#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# Adapted for GACRT dataset format.
#
import glob
import os
import struct
import numpy as np
import torch
from tqdm import tqdm
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud
import json
import pickle
from collections import defaultdict
from utils.od_loss_mask import build_od_loss_mask, transform_bboxes_pca

# GACRT014 default; override via config `lidar2ego` for other vehicles.
DEFAULT_LIDAR2EGO = np.array([[ -0.0290051,   -0.999232,   0.0266734,     1.10341 ],
                [ 0.999562,  -0.0291982, -0.00689323,  0.00806815 ],
                [ 0.00766665,   0.0264612,    0.999621,     1.93886 ],
                [ 0.0, 0.0, 0.0, 1.0 ]])
LIDAR2EGO = DEFAULT_LIDAR2EGO  # backward-compatible alias

_PCD_DTYPE = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
}


def get_lidar2ego(args=None, json_data=None):
    """Return 4x4 lidar-to-ego used only to place the panoramic RV camera.

    When points and poses are both already in the IMU frame
    (``pose_frame`` / ``point_frame`` == ``imu``), this must be identity —
    applying a lidar-top extrinsic would incorrectly rotate/translate the RV
    camera relative to the fused IMU cloud.

    For ``ego``-frame GACRT clips that still want the RV at lidar-top, keep
    using config ``lidar2ego`` (do not force identity).
    """
    pose_frame = None if json_data is None else json_data.get("pose_frame")
    point_frame = None if json_data is None else json_data.get("point_frame")
    if pose_frame is None and args is not None:
        pose_frame = getattr(args, "pose_frame", None)
    if point_frame is None and args is not None:
        point_frame = getattr(args, "point_frame", None)

    if pose_frame == "imu" and point_frame == "imu":
        return np.eye(4, dtype=np.float64)

    mat = None if args is None else getattr(args, "lidar2ego", None)
    if mat is None:
        return DEFAULT_LIDAR2EGO.copy()
    arr = np.array(mat, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"lidar2ego must be 4x4, got {arr.shape}")
    return arr


# KITTI-360 / GS-LiDAR axis remap: body (x-fwd,y-left,z-up) → RV camera.
BODY_TO_RV = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

# Fused PCD source channel. 0 is the top OT128 and is not a QT128 camera.
# Slots are stable raydrop indices even when a source is missing in a frame.
BLIND_LIDARS = (
    (1, "left"),
    (2, "right"),
    (3, "front"),
    (4, "back"),
)

# lidar-top-to-car is nearly a pure yaw: sensor x=left, y=back, z=up.
# Remap that housing frame to x-forward, y-left, z-up before BODY_TO_RV.
HESAI_TO_BODY = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def _find_calib_4x4(obj):
    if isinstance(obj, dict):
        data = obj.get("data")
        if obj.get("cols") == 4 and obj.get("rows") == 4 and isinstance(data, list):
            arr = np.asarray(data, dtype=np.float64)
            if arr.shape == (4, 4):
                return arr
        for value in obj.values():
            found = _find_calib_4x4(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_calib_4x4(value)
            if found is not None:
                return found
    return None


def load_calib_4x4(path):
    with open(path, "r") as f:
        mat = _find_calib_4x4(json.load(f))
    if mat is None:
        raise ValueError(f"No 4x4 sensor_calib in {path}")
    return mat


def load_blind_extrinsics(calib_dir):
    """Lidar-to-IMU for each blind QT128.

    Calib files are lidar-to-car and gnss-to-lidar-top. SLAM poses are IMU-to-world.
    IMU is treated as the GNSS frame:
        T_gnss_to_car = T_lidar_top_to_car @ T_gnss_to_lidar_top
        T_imu_from_lidar = inv(T_gnss_to_car) @ T_lidar_to_car
    """
    calib_dir = os.path.abspath(os.path.expanduser(str(calib_dir)))
    top_to_car = load_calib_4x4(os.path.join(calib_dir, "calib_lidar_top_to_car.json"))
    gnss_to_top = load_calib_4x4(os.path.join(calib_dir, "calib_gnss_to_lidar_top.json"))
    gnss_to_car = top_to_car @ gnss_to_top
    extrinsics = {}
    for source_id, name in BLIND_LIDARS:
        lidar_to_car = load_calib_4x4(
            os.path.join(calib_dir, f"calib_lidar_{name}_to_car.json"))
        imu_from_lidar = np.linalg.inv(gnss_to_car) @ lidar_to_car
        extrinsics[source_id] = {
            "name": name,
            "slot": source_id - 1,
            "imu_from_lidar": imu_from_lidar,
        }
        t = imu_from_lidar[:3, 3]
        print(f"[Blind calib] source {source_id} {name} "
              f"imu_t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})")
    return extrinsics


def _apply_transform(points, transform):
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    homo = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (homo @ np.asarray(transform, dtype=np.float64).T)[:, :3].astype(np.float32)


def _rt_from_w2c(w2c):
    rotation = np.transpose(w2c[:3, :3]).astype(np.float32).copy()
    translation = w2c[:3, 3].astype(np.float32).copy()
    return rotation, translation


def _backward_rt(rotation, translation):
    flip = np.array([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=np.float32).reshape(3, 3)
    return rotation @ flip, translation * np.array([-1, 1, -1], dtype=np.float32)


def _body_ignore_mask(body_cam, scene_cam, height, width, vfov, hfov):
    """1 where a body return is the only point on that pixel."""
    from utils.camera_utils import project_points_to_rv
    body_depth, _ = project_points_to_rv(body_cam, None, height, width, vfov, hfov)
    scene_depth, _ = project_points_to_rv(scene_cam, None, height, width, vfov, hfov)
    return ((body_depth > 0) & (scene_depth == 0)).astype(np.float32)


def build_blind_frame_cameras(
    points_imu, intensity, source, lidar2global, extrinsics, args,
    frame_idx, n_frames, timestamp, lidar_path,
):
    """One camera per blind lidar that has points in this frame.

    Depth uses that lidar's points only, expressed from the lidar origin.
    Body hits are removed from depth and marked ignore where they leave a hole.

    ``cam_num==1`` (qt128 default): a single 180° half with ``hfov=[-90,90]``.
    ``cam_num==2``: classic forward + backward halves stitched to 360°.
    """
    if source is None:
        raise RuntimeError("blind_lidar_origins requires a PCD source channel")

    finite = np.isfinite(points_imu).all(axis=1)
    points_imu = points_imu[finite]
    intensity = intensity[finite]
    source = np.asarray(source)[finite]
    near_r = getattr(args, "ego_near_radius", None)
    if near_r is not None and float(near_r) > 0:
        keep = np.linalg.norm(points_imu, axis=1) > float(near_r)
        points_imu = points_imu[keep]
        intensity = intensity[keep]
        source = source[keep]

    height, width = int(args.hw[0]), int(args.hw[1])
    aabb = getattr(args, "ego_body_aabb", None)
    n_sensors = len(BLIND_LIDARS)
    half_only = int(getattr(args, "cam_num", 2) or 2) == 1
    cameras = []
    init_chunks = []
    n_body = 0
    n_supervise = 0

    for source_id, _name in BLIND_LIDARS:
        selected = source == source_id
        if not np.any(selected):
            continue
        pts = points_imu[selected]
        inten = intensity[selected]
        is_body = points_in_aabb(pts, aabb) if aabb is not None else np.zeros(len(pts), dtype=bool)
        n_body += int(is_body.sum())
        scene_pts = pts[~is_body]
        scene_inten = inten[~is_body]
        body_pts = pts[is_body]
        if len(scene_pts) == 0 and len(body_pts) == 0:
            continue

        spec = extrinsics[source_id]
        world_from_lidar = lidar2global @ spec["imu_from_lidar"]
        w2c = BODY_TO_RV @ HESAI_TO_BODY @ np.linalg.inv(world_from_lidar)
        rotation, translation = _rt_from_w2c(w2c)

        scene_world = _apply_transform(scene_pts, lidar2global)
        body_world = _apply_transform(body_pts, lidar2global)
        if len(scene_world):
            init_chunks.append(scene_world)
            n_supervise += int(len(scene_world))

        def to_cam(pts_world, rot, trans):
            if len(pts_world) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return (pts_world @ rot + trans).astype(np.float32)

        scene_cam = to_cam(scene_world, rotation, translation)
        body_cam = to_cam(body_world, rotation, translation)
        mask_front = _body_ignore_mask(body_cam, scene_cam, height, width, args.vfov, args.hfov)
        if len(scene_inten) == 0:
            scene_inten = np.zeros((0,), dtype=np.float32)

        slot = spec["slot"]
        uid_front = slot * n_frames + frame_idx
        cameras.append(CameraInfo(
            uid=uid_front, R=rotation, T=translation, timestamp=timestamp,
            pointcloud_camera=scene_cam, intensity=scene_inten.astype(np.float32),
            towards="forward", lidar_path=lidar_path, sensor_id=slot,
            frame_idx=frame_idx, ego_body_mask=mask_front,
            ego_body_points_cam=body_cam if len(body_cam) else None,
            ego_body_intensity=inten[is_body].astype(np.float32) if np.any(is_body) else None,
        ))
        if half_only:
            continue

        rotation_back, translation_back = _backward_rt(rotation, translation)
        scene_cam_back = to_cam(scene_world, rotation_back, translation_back)
        body_cam_back = to_cam(body_world, rotation_back, translation_back)
        mask_back = _body_ignore_mask(
            body_cam_back, scene_cam_back, height, width, args.vfov, args.hfov)
        uid_back = uid_front + n_sensors * n_frames
        cameras.append(CameraInfo(
            uid=uid_back, R=rotation_back, T=translation_back, timestamp=timestamp,
            pointcloud_camera=scene_cam_back, intensity=scene_inten.astype(np.float32),
            towards="backward", lidar_path=lidar_path, sensor_id=slot,
            frame_idx=frame_idx, ego_body_mask=mask_back,
            ego_body_points_cam=body_cam_back if len(body_cam_back) else None,
            ego_body_intensity=inten[is_body].astype(np.float32) if np.any(is_body) else None,
        ))

    body_imu_chunks = []
    body_imu_inten = []
    for source_id, _name in BLIND_LIDARS:
        selected = source == source_id
        if not np.any(selected) or aabb is None:
            continue
        pts = points_imu[selected]
        inten_s = intensity[selected]
        is_body = points_in_aabb(pts, aabb)
        if np.any(is_body):
            body_imu_chunks.append(pts[is_body])
            body_imu_inten.append(inten_s[is_body])
    body_imu = (np.concatenate(body_imu_chunks, axis=0) if body_imu_chunks
                else np.zeros((0, 3), dtype=np.float32))
    body_imu_i = (np.concatenate(body_imu_inten, axis=0) if body_imu_inten
                  else np.zeros((0,), dtype=np.float32))

    if init_chunks:
        init_world = np.concatenate(init_chunks, axis=0)
    else:
        init_world = np.zeros((0, 3), dtype=np.float32)
    return cameras, init_world, n_body, n_supervise, body_imu, body_imu_i


def load_lidar_xyz_intensity(path, n_feats=5, return_source=False):
    """Load xyz + intensity from a GACRT .bin or binary / binary_compressed .pcd.

    .bin: float32 rows of (x, y, z, intensity, ring, ...).
    .pcd: FIELDS must include x, y, z; intensity/source are optional.
    """
    path = str(path)
    if path.lower().endswith(".pcd"):
        xyz, intensity, source = _load_pcd_xyz_intensity(path, return_source=True)
    else:
        points_raw = np.fromfile(path, dtype=np.float32)
        if points_raw.size % n_feats != 0:
            raise ValueError(
                f"Point cloud size {points_raw.size} is not divisible by n_feats={n_feats}: {path}")
        points_raw = points_raw.reshape(-1, n_feats)
        xyz = points_raw[:, :3].astype(np.float32)
        intensity = points_raw[:, 3].astype(np.float32)
        source = None
    if return_source:
        return xyz, intensity, source
    return xyz, intensity


def _lzf_decompress(data, expected_size):
    try:
        import lzf
    except ImportError as exc:
        raise ImportError(
            "binary_compressed PCD requires python-lzf. Install with: pip install python-lzf"
        ) from exc
    out = lzf.decompress(data, expected_size)
    if out is None or len(out) != expected_size:
        raise ValueError("LZF decompress failed or size mismatch")
    return out


def _load_pcd_xyz_intensity(path, return_source=False):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD (no DATA line): {path}")
            text = line.decode("ascii", errors="ignore").strip()
            header_lines.append(text)
            if text.startswith("DATA"):
                break
        meta = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
        data_mode = meta.get("DATA", "").split()[0]
        n = int(meta["POINTS"])
        fields = meta["FIELDS"].split()
        sizes = list(map(int, meta["SIZE"].split()))
        types = meta["TYPE"].split()
        raw = f.read()

    cols = {}
    if data_mode == "binary":
        dtype = []
        for name, sz, tp in zip(fields, sizes, types):
            key = (tp, sz)
            if key not in _PCD_DTYPE:
                raise ValueError(f"Unsupported PCD field {name} type={tp} size={sz} in {path}")
            dtype.append((name, _PCD_DTYPE[key]))
        arr = np.frombuffer(raw, dtype=np.dtype(dtype), count=n)
        for name in arr.dtype.names:
            cols[name] = np.ascontiguousarray(arr[name])
    elif data_mode == "binary_compressed":
        compressed_size, uncompressed_size = struct.unpack_from("<II", raw, 0)
        dec = _lzf_decompress(raw[8:8 + compressed_size], uncompressed_size)
        off = 0
        for name, sz, tp in zip(fields, sizes, types):
            key = (tp, sz)
            if key not in _PCD_DTYPE:
                raise ValueError(f"Unsupported PCD field {name} type={tp} size={sz} in {path}")
            nbytes = n * sz
            cols[name] = np.frombuffer(dec[off:off + nbytes], dtype=np.dtype(_PCD_DTYPE[key])).copy()
            off += nbytes
    else:
        raise ValueError(f"Only binary / binary_compressed PCD is supported, got '{data_mode}': {path}")

    if not {"x", "y", "z"}.issubset(cols):
        raise ValueError(f"PCD missing xyz fields: {path}")
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float32)
    if "intensity" in cols:
        intensity = cols["intensity"].astype(np.float32)
    else:
        intensity = np.zeros(len(xyz), dtype=np.float32)
    source = cols["source"].astype(np.int32) if "source" in cols else None
    if return_source:
        return xyz, intensity, source
    return xyz, intensity


def points_in_aabb(points, aabb):
    """Axis-aligned box test. aabb is [[xmin,xmax],[ymin,ymax],[zmin,zmax]] in the point frame."""
    box = np.asarray(aabb, dtype=np.float64)
    if box.shape != (3, 2):
        raise ValueError(f"ego_body_aabb must be 3x2 [[xmin,xmax],[ymin,ymax],[zmin,zmax]], got {box.shape}")
    return (
        (points[:, 0] >= box[0, 0]) & (points[:, 0] <= box[0, 1]) &
        (points[:, 1] >= box[1, 0]) & (points[:, 1] <= box[1, 1]) &
        (points[:, 2] >= box[2, 0]) & (points[:, 2] <= box[2, 1])
    )


def parse_input_lidar_sources(args):
    """Normalize config ``input_lidar_source`` to a list of ints, or None.

    Accepts: null | int | [int, ...]  (OmegaConf ListConfig OK).
    """
    raw = getattr(args, "input_lidar_source", None)
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return [int(raw)]
    # list / tuple / OmegaConf list
    try:
        return [int(x) for x in list(raw)]
    except TypeError as exc:
        raise ValueError(
            f"input_lidar_source must be null, int, or list of ints, got {type(raw)}: {raw}"
        ) from exc


def filter_lidar_sources(points, intensity, source, args):
    """Split training-input sources vs all-source supervision; drop 补盲 body hits.

    ``input_lidar_source`` may be a single id (e.g. 0=top) or a list
    (e.g. [0, 1, 2] = top + left/right 补盲) used for Gaussian init + raydrop.

    Returns:
        points, intensity, input_mask, n_body, source after near-radius + body filtering.
    """
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    intensity = intensity[finite]
    if source is not None:
        source = source[finite]

    near_r = getattr(args, "ego_near_radius", 2.5)
    if near_r is not None and float(near_r) > 0:
        keep = np.linalg.norm(points, axis=1) > float(near_r)
        points = points[keep]
        intensity = intensity[keep]
        if source is not None:
            source = source[keep]

    n_body = 0
    aabb = getattr(args, "ego_body_aabb", None)
    body_sources = getattr(args, "ego_body_filter_sources", None)
    if aabb is not None and source is not None and body_sources:
        body_src = np.asarray(body_sources, dtype=np.int32)
        is_fill = np.isin(source, body_src)
        is_body = is_fill & points_in_aabb(points, aabb)
        n_body = int(is_body.sum())
        keep = ~is_body
        points = points[keep]
        intensity = intensity[keep]
        source = source[keep]

    input_ids = parse_input_lidar_sources(args)
    if input_ids is not None and source is not None:
        input_mask = np.isin(source, np.asarray(input_ids, dtype=np.int32))
    else:
        input_mask = np.ones(len(points), dtype=bool)
    return points, intensity, input_mask, n_body, source

class NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that handles numpy version mismatch between save and load.

    Numpy >= 1.26 uses ``numpy._core`` as the internal module path,
    while older versions use ``numpy.core``. When a pkl saved with
    numpy 1.26+ is loaded in an environment with numpy < 1.26,
    pickle fails with ``ModuleNotFoundError: No module named 'numpy._core'``.
    This class remaps ``numpy._core`` → ``numpy.core`` (and vice versa)
    so the pkl can be loaded regardless of the numpy version.
    """
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module.startswith('numpy.core'):
            module = module.replace('numpy.core', 'numpy._core')
        return super().find_class(module, name)


def load_pkl_compat(path_or_file):
    """Load a pickle file with numpy version compatibility.

    Args:
        path_or_file: Either a file path (str/PathLike) or an already-opened
            binary file object.
    """
    if isinstance(path_or_file, (str, os.PathLike)):
        with open(path_or_file, 'rb') as f:
            return NumpyCompatUnpickler(f).load()
    else:
        return NumpyCompatUnpickler(path_or_file).load()


def iter_pkl_frames(pkl_data):
    """遍历 pkl 中所有唯一帧（关键帧 + sweep），与 preprocess/gacrt_to_gs.py 展开逻辑一致。

    Yields:
        dict with keys: timestamp, ego2global, instances, lidar_path, pkl_type, keyframe_idx
    """
    seen_paths = set()
    for kf_idx, item in enumerate(pkl_data):
        ego2global = item["ego2global"]
        kf_instances = item.get("instances", [])

        kf_path = item["lidar_points"]["lidar_path"]
        if kf_path not in seen_paths:
            seen_paths.add(kf_path)
            yield {
                "timestamp": item["timestamp"],
                "ego2global": ego2global,
                "instances": kf_instances,
                "lidar_path": kf_path,
                "pkl_type": "keyframe",
                "keyframe_idx": kf_idx,
            }

        for sweep in item.get("lidar_sweeps", []):
            sp = sweep["lidar_points"]
            sp_path = sp["lidar_path"]
            if sp_path in seen_paths:
                continue
            seen_paths.add(sp_path)
            if "instances" in sweep:
                yield {
                    "timestamp": sweep["timestamp"],
                    "ego2global": sp["ego2global"],
                    "instances": sweep["instances"],
                    "lidar_path": sp_path,
                    "pkl_type": "sweep",
                    "keyframe_idx": kf_idx,
                }
            else:
                # 少数早于关键帧的 sweep 无独立标注，回退到所属关键帧
                yield {
                    "timestamp": sweep["timestamp"],
                    "ego2global": ego2global,
                    "instances": kf_instances,
                    "lidar_path": sp_path,
                    "pkl_type": "sweep",
                    "keyframe_idx": kf_idx,
                }


def extract_od_annotations(instances, ego2global, moving_obj_ids):
    """从 instances 提取移动目标的世界坐标系 bbox 列表。"""
    annotations = []
    for inst in instances:
        obj_id = inst["obj_ids"]
        if obj_id not in moving_obj_ids:
            continue
        bbox_world = transform_bbox_to_world(inst["bbox_3d"], ego2global)
        annotations.append((obj_id, bbox_world))
    return annotations


def build_od_annotations_by_lidar_path(pkl_data, moving_obj_ids):
    """按 lidar_path 预计算每帧 OD 标注（关键帧 + sweep）。"""
    od_by_path = {}
    for frame in iter_pkl_frames(pkl_data):
        od_by_path[frame["lidar_path"]] = extract_od_annotations(
            frame["instances"], frame["ego2global"], moving_obj_ids)
    return od_by_path


def identify_moving_objects(pkl_data, dynamic_labels, min_displacement=0.5, min_speed=None):
    """通过 obj_id 追踪识别真正移动中的目标。

    对同一 obj_id 的 bbox 中心在不同关键帧之间做线性拟合求速度。
    - 若指定 min_speed：仅当速度模长 > min_speed (m/s) 时视为移动目标。
    - 否则：使用最大位移 > min_displacement (m) 判定（兼容旧逻辑）。

    Args:
        pkl_data: pkl 加载的数据 (list of dicts)
        dynamic_labels: 候选动态类别集合, e.g. {0, 1, 2, 5, 7}
        min_displacement: 最小位移阈值 (米), min_speed 未设置时生效
        min_speed: 最小速度阈值 (m/s), 优先于 min_displacement

    Returns:
        moving_obj_ids: set, 移动目标的 obj_id 集合
        obj_velocities: dict, obj_id → velocity (3,) 世界坐标系 (m/s)
    """
    # 收集每个 obj_id 的轨迹: [(timestamp, center_world)]
    obj_trajectories = defaultdict(list)

    for frame in iter_pkl_frames(pkl_data):
        ego2global = frame["ego2global"]
        timestamp = frame["timestamp"]
        for inst in frame["instances"]:
            label = inst["bbox_label_3d"]
            if label not in dynamic_labels:
                continue
            obj_id = inst["obj_ids"]
            bbox_3d = inst["bbox_3d"]  # [cx, cy, cz, w, l, h, yaw] ego frame
            center_ego = np.array([bbox_3d[0], bbox_3d[1], bbox_3d[2], 1.0])
            center_world = (ego2global @ center_ego)[:3]
            obj_trajectories[obj_id].append((timestamp, center_world))

    moving_obj_ids = set()
    obj_velocities = {}

    for obj_id, traj in obj_trajectories.items():
        if len(traj) < 2:
            continue
        traj.sort(key=lambda x: x[0])
        centers = np.array([c for _, c in traj])
        ts = np.array([t for t, _ in traj])
        t_mean = ts.mean()
        dt = ts - t_mean
        c_mean = centers.mean(axis=0)
        dc = centers - c_mean
        v = np.sum(dt[:, None] * dc, axis=0) / (np.sum(dt ** 2) + 1e-8)
        speed = float(np.linalg.norm(v))
        if min_speed is not None:
            is_moving = speed > min_speed
        else:
            max_disp = np.max(np.linalg.norm(centers - centers[0], axis=1))
            is_moving = max_disp > min_displacement
        if is_moving:
            moving_obj_ids.add(obj_id)
            obj_velocities[obj_id] = v

    if min_speed is not None:
        threshold_msg = f"Min speed threshold: {min_speed}m/s"
    else:
        threshold_msg = f"Min displacement threshold: {min_displacement}m"
    print(f"[OD] Total tracked objects: {len(obj_trajectories)}, "
          f"Moving objects: {len(moving_obj_ids)}, "
          f"{threshold_msg}")
    return moving_obj_ids, obj_velocities


def transform_bbox_to_world(bbox_3d, ego2global):
    """将 3D bbox 从 ego 坐标系变换到世界坐标系。

    Args:
        bbox_3d: (7,) [cx, cy, cz, w, l, h, yaw] (ego 坐标系)
        ego2global: (4, 4) 变换矩阵

    Returns:
        dict: {center: (3,), size: (3,), yaw: float} (世界坐标系)
    """
    center_ego = np.array([bbox_3d[0], bbox_3d[1], bbox_3d[2], 1.0])
    center_world = (ego2global @ center_ego)[:3]
    size = np.array([bbox_3d[3], bbox_3d[4], bbox_3d[5]])
    # 提取 ego2global 的偏航角
    yaw_offset = np.arctan2(ego2global[1, 0], ego2global[0, 0])
    yaw_world = bbox_3d[6] + yaw_offset
    return {'center': center_world, 'size': size, 'yaw': yaw_world}


def points_in_rotated_3d_bbox(points, bboxes):
    """检测点是否在旋转3D bbox内。

    Args:
        points: (N, 3) 世界坐标系下的点
        bboxes: list of dict, 每个包含 center, size, yaw

    Returns:
        obj_ids_per_point: (N,) int array, 每个点所属的 obj_id 索引 (>=0), -1 表示不属于任何 bbox
        bbox_list: list, 与 obj_ids_per_point 中的索引对应的 bbox 列表
    """
    N = len(points)
    obj_ids_per_point = np.full(N, -1, dtype=np.int32)
    bbox_list = []

    for bbox_idx, bbox in enumerate(bboxes):
        center = bbox['center']
        size = bbox['size']
        yaw = bbox['yaw']

        # 变换到 bbox 局部坐标系
        dx = points[:, 0] - center[0]
        dy = points[:, 1] - center[1]
        dz = points[:, 2] - center[2]

        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        inside = (np.abs(local_x) < size[0] / 2) & \
                 (np.abs(local_y) < size[1] / 2) & \
                 (np.abs(dz) < size[2] / 2)

        # 标记属于此 bbox 的点（后续 bbox 可覆盖之前的）
        obj_ids_per_point[inside] = bbox_idx
        bbox_list.append(bbox)

    return obj_ids_per_point, bbox_list


def readGacrtInfo(args):
    """Read GACRT dataset and return SceneInfo.

    Expected directory structure:
        data/
        ├── GACRT014_1740908194.pkl                # annotation file
        ├── GACRT014_1740908194/
        │   └── transforms_GACRT014_1740908194_all.json  # preprocessed transforms
        └── raw_data/
            └── GACRT014_1740908194/
                └── lidar/                         # point cloud .bin files
                    ├── 1740908196400000000_640_19.bin
                    └── ...

    Key assumptions about GACRT / PCI data:
        - points are in the ego / IMU / vehicle-body frame
        - lidar2world in the JSON is ego2global (or IMU-to-world)
        - lidar2ego places the panoramic RV camera; forced to identity when
          pose_frame=point_frame=imu (fused IMU clouds). ego-frame clips may
          still use a lidar-top extrinsic.
        - Point cloud: .bin float32 or binary / binary_compressed .pcd
        - Optional `source` channel: ``input_lidar_source`` (int or list) selects
          which ids go into Gaussian init + raydrop; remaining points still
          supervise depth after body filtering
        - ego_body_aabb + ego_body_filter_sources drop 补盲 hits on the car body
        - LiDAR FoV: vfov=[-25, 15], hfov=[-180, 180]

    The JSON file may contain either keyframes only (40 frames) or all sweeps
    expanded (249 frames), depending on the preprocessing mode.
    """
    path = args.source_path
    eval_mode = args.eval
    num_pts = args.num_pts
    time_duration = args.time_duration
    debug_cuda = args.debug_cuda

    use_od_labels = getattr(args, 'use_od_labels', False)

    assert args.vfov is not None and args.hfov is not None

    sequence_id = args.sequence_id

    # Load transforms JSON (generated by preprocess/gacrt_to_gs.py)
    json_path = os.path.join(path, sequence_id, f"transforms_{sequence_id}_all.json")
    with open(json_path, "r") as f:
        json_data = json.load(f)

    lidar2ego = get_lidar2ego(args, json_data)
    if np.allclose(lidar2ego, np.eye(4)):
        print(f"[Frame] pose_frame={json_data.get('pose_frame')} "
              f"point_frame={json_data.get('point_frame')} → lidar2ego=I "
              f"(skip ego2lidar for RV camera)")
    else:
        t = lidar2ego[:3, 3]
        print(f"[Frame] pose_frame={json_data.get('pose_frame')} "
              f"point_frame={json_data.get('point_frame')} → "
              f"lidar2ego t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})")

    poses = json_data["frames"]
    val_frame_ids = set(json_data.get("val_frame_ids", []))

    frames = len(poses)
    args.frames = frames

    # Validate val_frame_ids are within valid range
    for vf in val_frame_ids:
        assert 0 <= vf < frames, \
            f"val_frame_id {vf} out of range [0, {frames}), check preprocessing"

    # --- OD annotation loading ---
    moving_obj_ids = set()
    obj_velocities = {}
    # lidar_path → list of OD annotations (in world frame)
    od_annotations_by_path = {}

    if use_od_labels:
        # Load pkl for OD annotations
        pkl_path = getattr(args, 'pkl_path', None)
        if pkl_path is None:
            pkl_candidates = glob.glob(os.path.join(path, f"{sequence_id}.pkl"))
            if pkl_candidates:
                pkl_path = pkl_candidates[0]
            else:
                print(f"[OD Warning] pkl file not found, OD labels disabled")
                use_od_labels = False

    if use_od_labels:
        pkl_data = load_pkl_compat(pkl_path)
        dynamic_labels = set(getattr(args, 'od_dynamic_labels', [0, 1, 2, 5, 7]))
        min_disp = getattr(args, 'od_min_displacement', 0.5)
        min_speed = getattr(args, 'od_min_speed', None)

        # Identify moving objects via tracking
        moving_obj_ids, obj_velocities = identify_moving_objects(
            pkl_data, dynamic_labels, min_disp, min_speed=min_speed)

        # Pre-compute world-frame bboxes for each unique frame (keyframe + sweep)
        od_annotations_by_path = build_od_annotations_by_lidar_path(
            pkl_data, moving_obj_ids)

        n_sweep_frames = sum(
            1 for frame in iter_pkl_frames(pkl_data) if frame["pkl_type"] == "sweep")
        print(f"[OD] Loaded annotations for {len(od_annotations_by_path)} unique frames "
              f"({len(pkl_data)} keyframes + {n_sweep_frames} sweeps), "
              f"{len(moving_obj_ids)} moving objects tracked")

    point_list = []
    points_time = []
    points_is_dynamic = []   # per-point dynamic flag
    points_obj_velocity = [] # per-point init velocity
    cam_infos = []
    n_body_total = 0
    n_input_total = 0
    n_supervise_total = 0
    body_imu_chunks = []
    body_imu_inten_chunks = []
    blind_origins = bool(getattr(args, "blind_lidar_origins", False))
    blind_extrinsics = None
    if blind_origins:
        if use_od_labels:
            raise RuntimeError("blind_lidar_origins does not support use_od_labels")
        calib_dir = getattr(args, "calib_dir", None)
        if not calib_dir:
            raise RuntimeError("blind_lidar_origins requires calib_dir")
        blind_extrinsics = load_blind_extrinsics(calib_dir)
        args.n_lidar_sensors = len(BLIND_LIDARS)
        print(f"[Blind] per-lidar cameras for sources "
              f"{[sid for sid, _ in BLIND_LIDARS]} (skip a source when a frame has no points)")

    for frame_idx in tqdm(range(frames), desc="Reading GacrtInfo"):
        pose_info = poses[frame_idx]
        lidar2global =  np.array(pose_info["lidar2world"])
        timestamp_real = pose_info["timestamp"]

        # Validate lidar2global is a valid rigid transform
        det = np.linalg.det(lidar2global[:3, :3])
        assert abs(abs(det) - 1.0) < 0.05, \
            f"Frame {frame_idx}: lidar2world is not a rigid transform (det={det:.4f})"

        # Get lidar file path — either from JSON (sweep-expanded mode)
        # or from pkl (keyframes-only mode)
        if "lidar_path" in pose_info:
            lidar_path = pose_info["lidar_path"]
            n_feats = pose_info.get("num_pts_feats", 5)
        else:
            # Fallback: read from pkl for keyframes-only mode
            pkl_path = getattr(args, 'pkl_path', None)
            if pkl_path is None:
                pkl_candidates = glob.glob(os.path.join(path, f"{sequence_id}.pkl"))
                if not pkl_candidates:
                    raise FileNotFoundError(
                        f"Cannot find pkl file for sequence {sequence_id}. "
                        f"Please specify pkl_path in config or place {sequence_id}.pkl in {path}/"
                    )
                pkl_path = pkl_candidates[0]

            pkl_data = load_pkl_compat(pkl_path)
            pkl_idx = pose_info["idx"]
            lidar_path = pkl_data[pkl_idx]["lidar_points"]["lidar_path"]
            n_feats = pkl_data[pkl_idx]["lidar_points"]["num_pts_feats"]

        full_lidar_path = os.path.join(path, lidar_path)

        if not os.path.exists(full_lidar_path):
            print(f"[Warning] Point cloud not found: {full_lidar_path}, skipping frame {frame_idx}")
            continue

        points, intensity, source = load_lidar_xyz_intensity(
            full_lidar_path, n_feats, return_source=True)
        # Normalize intensity to [0, 1] — GACRT raw range is [0, 255],
        # but Gaussian model uses sigmoid activation which outputs [0, 1].
        # Without normalization, the L1 loss compares [0,255] vs [0,1] directly.
        intensity = intensity / 255.0

        if blind_origins:
            t = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / max(frames - 1, 1)
            (blind_cams, init_points, n_body, n_sup,
             body_imu, body_imu_i) = build_blind_frame_cameras(
                points, intensity, source, lidar2global, blind_extrinsics, args,
                frame_idx, frames, t, lidar_path)
            n_body_total += n_body
            if body_imu is not None and len(body_imu):
                body_imu_chunks.append(body_imu)
                body_imu_inten_chunks.append(body_imu_i)
            if len(init_points) == 0 and not blind_cams:
                print(f"[Warning] No blind-lidar points: {full_lidar_path}, skipping frame {frame_idx}")
                if debug_cuda and frame_idx >= 15:
                    break
                continue
            if len(init_points):
                n_input_total += int(init_points.shape[0])
                n_supervise_total += int(n_sup)
                point_list.append(init_points)
                points_time.append(np.full((len(init_points), 1), t, dtype=np.float32))
                points_is_dynamic.append(np.zeros((len(init_points), 1), dtype=np.float32))
                points_obj_velocity.append(np.zeros((len(init_points), 3), dtype=np.float32))
            cam_infos.extend(blind_cams)
            if debug_cuda and frame_idx >= 15:
                break
            continue

        points, intensity, input_mask, n_body, _source = filter_lidar_sources(
            points, intensity, source, args)
        n_body_total += n_body
        if points.shape[0] == 0 or not input_mask.any():
            print(f"[Warning] No usable points after source/body filter: {full_lidar_path}, "
                  f"skipping frame {frame_idx}")
            continue

        # Transform points to world coordinates: p_world = p_ego/imu @ lidar2global.T
        points_homo = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
        points = (points_homo @ lidar2global.T)[:, :3]
        init_points = points[input_mask]
        init_intensity = intensity[input_mask]
        n_input_total += int(init_points.shape[0])
        n_supervise_total += int(points.shape[0])
        point_list.append(init_points)

        # --- OD: mark dynamic points and assign init velocity (on training-input points) ---
        n_pts = init_points.shape[0]
        is_dynamic_frame = np.zeros((n_pts, 1), dtype=np.float32)
        init_velocity_frame = np.zeros((n_pts, 3), dtype=np.float32)
        is_dynamic_all = np.zeros((points.shape[0], 1), dtype=np.float32)

        if use_od_labels and od_annotations_by_path:
            annotations = od_annotations_by_path.get(lidar_path, [])

            if annotations:
                bboxes = [bbox for _, bbox in annotations]
                obj_id_list = [oid for oid, _ in annotations]

                obj_ids_per_point, _ = points_in_rotated_3d_bbox(init_points, bboxes)
                dynamic_mask = obj_ids_per_point >= 0
                is_dynamic_frame[dynamic_mask] = 1.0

                obj_ids_all, _ = points_in_rotated_3d_bbox(points, bboxes)
                is_dynamic_all[obj_ids_all >= 0] = 1.0

                if getattr(args, 'od_velocity_init', True):
                    for local_idx, obj_id in enumerate(obj_id_list):
                        if obj_id in obj_velocities:
                            point_mask = obj_ids_per_point == local_idx
                            init_velocity_frame[point_mask] = obj_velocities[obj_id]
        points_is_dynamic.append(is_dynamic_frame)
        points_obj_velocity.append(init_velocity_frame)

        # Timestamp for dynamic scene: linearly map frame index to time_duration
        t = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / max(frames - 1, 1)
        point_time = np.full_like(init_points[:, :1], t)
        points_time.append(point_time)

        idx = frame_idx

        # World → body/ego/IMU (via lidar2ego; identity when cloud already in that frame)
        # then body → GS panoramic camera axes. R stored transposed for glm/CUDA.
        w2l = BODY_TO_RV @ np.linalg.inv(lidar2ego) @ np.linalg.inv(lidar2global)
        R = np.transpose(w2l[:3, :3])
        T = w2l[:3, 3]
        points_cam = points @ R + T
        points_cam_input = init_points @ R + T

        # Physical RV strip: drop moving-object points so they never enter pts_depth.
        if getattr(args, 'od_mask_gt_loss', False) and is_dynamic_all.any():
            static_sel = is_dynamic_all[:, 0] < 0.5
            points_cam_rv = points_cam[static_sel]
            intensity_rv = intensity[static_sel]
        else:
            points_cam_rv = points_cam
            intensity_rv = intensity
        if getattr(args, 'od_mask_gt_loss', False) and is_dynamic_frame.any():
            static_in = is_dynamic_frame[:, 0] < 0.5
            points_cam_in = points_cam_input[static_in]
            intensity_in = init_intensity[static_in]
        else:
            points_cam_in = points_cam_input
            intensity_in = init_intensity
        if parse_input_lidar_sources(args) is None:
            points_cam_in = None
            intensity_in = None

        # Forward half (front 180 degrees)
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T,
                                    timestamp=t, pointcloud_camera=points_cam_rv, intensity=intensity_rv,
                                    towards='forward', lidar_path=lidar_path,
                                    pointcloud_camera_input=points_cam_in, intensity_input=intensity_in,
                                    sensor_id=0, frame_idx=frame_idx))

        # Backward half (back 180 degrees)
        R_back = R @ np.array([-1, 0, 0,
                               0, 1, 0,
                               0, 0, -1]).reshape(3, 3)
        T_back = T * np.array([-1, 1, -1])
        points_cam_back = points @ R_back + T_back
        points_cam_back_in = init_points @ R_back + T_back
        if getattr(args, 'od_mask_gt_loss', False) and is_dynamic_all.any():
            points_cam_back = points_cam_back[static_sel]
        if getattr(args, 'od_mask_gt_loss', False) and is_dynamic_frame.any():
            points_cam_back_in = points_cam_back_in[static_in]
        if parse_input_lidar_sources(args) is None:
            points_cam_back_in = None
        cam_infos.append(CameraInfo(uid=idx + frames, R=R_back, T=T_back,
                                    timestamp=t, pointcloud_camera=points_cam_back, intensity=intensity_rv,
                                    towards='backward', lidar_path=lidar_path,
                                    pointcloud_camera_input=points_cam_back_in, intensity_input=intensity_in,
                                    sensor_id=0, frame_idx=frame_idx))

        if debug_cuda and frame_idx >= 15:
            break

    if parse_input_lidar_sources(args) is not None or getattr(args, "ego_body_aabb", None) is not None:
        n_frames_kept = max(len(point_list), 1)
        print(f"[LiDAR source] input_source={parse_input_lidar_sources(args)} "
              f"init_pts={n_input_total} supervise_pts={n_supervise_total} "
              f"ego_body_removed={n_body_total} "
              f"(avg body/frame={n_body_total / n_frames_kept:.1f})")

    if body_imu_chunks:
        from utils.ego_body_mask import save_ego_body_cloud
        body_xyz = np.concatenate(body_imu_chunks, axis=0)
        body_i = np.concatenate(body_imu_inten_chunks, axis=0)
        body_path = os.path.join(args.model_path, "ego_body_imu.npz")
        save_ego_body_cloud(body_path, body_xyz, body_i)
        print(f"[Ego body] saved rigid template {body_path} n={len(body_xyz)}")

    if len(point_list) == 0:
        raise RuntimeError("No point cloud files found! Please check that raw .bin/.pcd files exist.")

    pointcloud = np.concatenate(point_list, axis=0)
    assert pointcloud.shape[0] > 0, \
        "Concatenated point cloud is empty — all .bin/.pcd files may have been skipped"
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    pointcloud_is_dynamic = np.concatenate(points_is_dynamic, axis=0)
    pointcloud_init_velocity = np.concatenate(points_obj_velocity, axis=0)

    if getattr(args, 'od_mask_init', False):
        static_sel = pointcloud_is_dynamic[:, 0] < 0.5
        n_before = pointcloud.shape[0]
        pointcloud = pointcloud[static_sel]
        pointcloud_timestamp = pointcloud_timestamp[static_sel]
        pointcloud_is_dynamic = pointcloud_is_dynamic[static_sel]
        pointcloud_init_velocity = pointcloud_init_velocity[static_sel]
        print(f"[OD] od_mask_init: kept {pointcloud.shape[0]}/{n_before} static init points")

    num_pts = min(num_pts, pointcloud.shape[0])
    indices = np.random.choice(pointcloud.shape[0], num_pts, replace=False)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]
    pointcloud_is_dynamic = pointcloud_is_dynamic[indices]
    pointcloud_init_velocity = pointcloud_init_velocity[indices]

    if use_od_labels:
        dynamic_ratio = pointcloud_is_dynamic.mean()
        print(f"[OD] Point cloud stats: {pointcloud.shape[0]} total, "
              f"{int(pointcloud_is_dynamic.sum())} dynamic ({dynamic_ratio:.2%}), "
              f"{int((1 - pointcloud_is_dynamic).sum())} static")

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))

    pca_path = os.path.join(args.model_path, 'transform_poses_pca.npz')
    reuse_pca = args.test_only or getattr(args, 'start_checkpoint', None) or os.path.exists(pca_path)
    if reuse_pca and os.path.exists(pca_path):
        data_npz = np.load(pca_path)
        transform = data_npz['transform']
        scale_factor = data_npz['scale_factor'].item()
        c2ws = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2ws)
        c2ws[:, :3, 3] *= scale_factor
        print(f"[PCA] Reused existing transform from {pca_path} (scale_factor={scale_factor})")
    else:
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, args.dynamic)
        np.savez(pca_path, transform=transform, scale_factor=scale_factor)
        c2ws = pad_poses(c2ws)

    for idx, cam_info in enumerate(cam_infos):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
        if cam_info.pointcloud_camera_input is not None:
            cam_info.pointcloud_camera_input[:] *= scale_factor
        if cam_info.ego_body_points_cam is not None and len(cam_info.ego_body_points_cam):
            cam_info.ego_body_points_cam[:] *= scale_factor

    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    # Transform init_velocity to PCA-aligned coordinate system (rotation only)
    if pointcloud_init_velocity is not None and np.any(pointcloud_init_velocity != 0):
        pointcloud_init_velocity = pointcloud_init_velocity @ transform[:3, :3].T
    args.scale_factor = float(scale_factor)

    od_mask_gt_loss = getattr(args, 'od_mask_gt_loss', False)
    od_bbox_scale = getattr(args, 'od_bbox_scale', 1.15)
    if od_mask_gt_loss and od_annotations_by_path:
        h_base, w_base = args.hw[0], args.hw[1]
        updated_cam_infos = []
        for cam_info in cam_infos:
            lp = cam_info.lidar_path
            raw_bboxes = [b for _, b in od_annotations_by_path.get(lp, [])]
            bboxes_pca = transform_bboxes_pca(raw_bboxes, transform, scale_factor)
            od_mask = build_od_loss_mask(
                cam_info.R, cam_info.T, bboxes_pca,
                h_base, w_base, args.vfov, args.hfov, od_bbox_scale,
            )
            updated_cam_infos.append(cam_info._replace(od_loss_mask=od_mask))
        cam_infos = updated_cam_infos
        coverages = [float(1.0 - m.mean()) for m in (c.od_loss_mask for c in cam_infos)]
        mean_cov = float(np.mean(coverages)) if coverages else 0.0
        max_cov = float(np.max(coverages)) if coverages else 0.0
        print(f"[OD] od_mask_gt_loss: stripped dynamic points from RV + "
              f"filled silhouette masks for {len(cam_infos)} camera views "
              f"(OD coverage mean={mean_cov:.2%} max={max_cov:.2%})")
        if mean_cov > 0.40:
            print("[OD] WARNING: OD silhouette covers >40% of the RV. "
                  "Check bbox PCA transform / od_min_speed; supervision is nearly empty.")

    mod = args.cam_num  # 1 = single 180° half; 2 = forward/backward stitch

    # Validate camera list length against cam_num (pairing for cam_num>=2)
    assert len(cam_infos) % mod == 0, \
        f"cam_infos length {len(cam_infos)} not divisible by cam_num={mod}, forward/backward pairing broken"

    # val_frame_ids stores local frame indices
    if blind_origins:
        train_cam_infos = [c for c in cam_infos if c.frame_idx not in val_frame_ids]
        test_cam_infos = [c for c in cam_infos if c.frame_idx in val_frame_ids]
        from utils.blind_dead_mask import dead_mask_enabled, build_dead_masks, attach_dead_masks
        if dead_mask_enabled(args):
            # Statistics from train frames only; the mask is a sensor property shared with test.
            dead, ignore = build_dead_masks(train_cam_infos, args)
            train_cam_infos = attach_dead_masks(train_cam_infos, dead, ignore)
            test_cam_infos = attach_dead_masks(test_cam_infos, dead, ignore)
        present = sorted({c.sensor_id for c in cam_infos})
        print(f"[Blind] cameras={len(cam_infos)} sensor_slots={present} "
              f"train={len(train_cam_infos)} test={len(test_cam_infos)}")
    elif eval_mode:
        train_cam_infos = [c for i, c in enumerate(cam_infos)
                           if (i // mod) not in val_frame_ids]
        test_cam_infos = [c for i, c in enumerate(cam_infos)
                          if (i // mod) in val_frame_ids]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = [c for i, c in enumerate(cam_infos)
                          if (i // mod) in val_frame_ids]

    # Validate test cameras are present (otherwise evaluation will silently produce nothing)
    # Skip this check in debug_cuda mode since it truncates frames before val_frame_ids
    if not debug_cuda:
        assert len(test_cam_infos) > 0, \
            "No test cameras — val_frame_ids may be empty or misconfigured"

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1

    ply_path = os.path.join(args.model_path, "points3d.ply")
    if not args.test_only:
        rgbs = np.random.random((pointcloud.shape[0], 3)) * 255.0
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp, pointcloud_is_dynamic)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # Attach init_velocity to BasicPointCloud (not stored in PLY, passed separately)
    if pcd is not None and pointcloud_init_velocity is not None:
        pcd = BasicPointCloud(
            points=pcd.points, colors=pcd.colors, normals=pcd.normals,
            time=pcd.time, is_dynamic=pcd.is_dynamic,
            init_velocity=pointcloud_init_velocity
        )

    time_interval = (time_duration[1] - time_duration[0]) / max(frames - 1, 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval)

    return scene_info


# ---- Pose normalization utilities (same as kitti360_loader) ----

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_scale_factor=True):
    """Transforms poses so principal components lie on XYZ axes.

    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
        fix_scale_factor: if True, use fixed scale 1/10 (static); else adaptive (dynamic).

    Returns:
        A tuple (poses, transform, scale_factor).
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    if fix_scale_factor:
        scale_factor = 1 / 10
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor
