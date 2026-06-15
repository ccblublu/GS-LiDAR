#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# Adapted for GACRT dataset format.
#
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud
import json
import pickle
from collections import defaultdict

LIDAR2EGO = np.array([[ -0.0290051,   -0.999232,   0.0266734,     1.10341 ],
                [ 0.999562,  -0.0291982, -0.00689323,  0.00806815 ],
                [ 0.00766665,   0.0264612,    0.999621,     1.93886 ],
                [ 0.0, 0.0, 0.0, 1.0 ]])

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


def identify_moving_objects(pkl_data, dynamic_labels, min_displacement=0.5):
    """通过 obj_id 追踪识别真正移动中的目标。

    对同一 obj_id 的 bbox 中心在不同关键帧之间计算最大位移，
    仅当最大位移超过阈值时才认为是移动目标（停放车辆不会被标记为动态）。

    Args:
        pkl_data: pkl 加载的数据 (list of dicts)
        dynamic_labels: 候选动态类别集合, e.g. {0, 1, 2, 5, 7}
        min_displacement: 最小位移阈值 (米), 超过此值才认为是移动目标

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
        # 计算最大位移（相对于首个观测位置）
        max_disp = np.max(np.linalg.norm(centers - centers[0], axis=1))
        if max_disp > min_displacement:
            moving_obj_ids.add(obj_id)
            # 线性回归求速度: center(t) = c_mean + v * (t - t_mean)
            ts = np.array([t for t, _ in traj])
            t_mean = ts.mean()
            dt = ts - t_mean
            c_mean = centers.mean(axis=0)
            dc = centers - c_mean
            v = np.sum(dt[:, None] * dc, axis=0) / (np.sum(dt ** 2) + 1e-8)
            obj_velocities[obj_id] = v

    print(f"[OD] Total tracked objects: {len(obj_trajectories)}, "
          f"Moving objects: {len(moving_obj_ids)}, "
          f"Min displacement threshold: {min_displacement}m")
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

    Key assumptions about GACRT data:
        - lidar2ego = Identity (points are in ego coordinate frame)
        - lidar2global = ego2global
        - Point cloud format: (x, y, z, intensity, ring_idx) as float32
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

        # Identify moving objects via tracking
        moving_obj_ids, obj_velocities = identify_moving_objects(
            pkl_data, dynamic_labels, min_disp)

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

        # GACRT format: 5 features per point (x, y, z, intensity, ring_idx)
        points_raw = np.fromfile(full_lidar_path, dtype=np.float32)
        points_raw = points_raw.reshape(-1, n_feats)

        # Validate point cloud feature count matches expected
        assert points_raw.shape[1] == n_feats, \
            f"Point cloud feature count mismatch: expected {n_feats}, got {points_raw.shape[1]}"

        points = points_raw[:, :3]  # x, y, z
        intensity = points_raw[:, 3]  # intensity
        # Normalize intensity to [0, 1] — GACRT raw range is [0, 255],
        # but Gaussian model uses sigmoid activation which outputs [0, 1].
        # Without normalization, the L1 loss compares [0,255] vs [0,1] directly.
        intensity = intensity / 255.0
        # ring_idx = points_raw[:, 4]  # not used in GS-LiDAR

        # Remove ego vehicle points (close to origin)
        # GACRT uses ego coordinate frame, so ego vehicle is at origin
        condition = np.linalg.norm(points, axis=1) > 2.5
        points = points[condition]
        intensity = intensity[condition]

        # Transform points to world coordinates: p_world = p_ego @ lidar2global.T
        points_homo = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
        # lidar2global_ = np.linalg.inv(LIDAR2EGO) @ lidar2global
        points = (points_homo @ lidar2global.T)[:, :3]
        point_list.append(points)

        # --- OD: mark dynamic points and assign init velocity ---
        n_pts = points.shape[0]
        is_dynamic_frame = np.zeros((n_pts, 1), dtype=np.float32)
        init_velocity_frame = np.zeros((n_pts, 3), dtype=np.float32)

        if use_od_labels and od_annotations_by_path:
            # Look up per-frame OD by lidar_path (sweep has its own annotations)
            frame_lidar_path = pose_info.get("lidar_path")
            if frame_lidar_path is None and "idx" in pose_info:
                pkl_item = pkl_data[pose_info["idx"]]
                frame_lidar_path = pkl_item["lidar_points"]["lidar_path"]
            annotations = od_annotations_by_path.get(frame_lidar_path, [])

            if annotations:
                # Collect world-frame bboxes for moving objects only
                bboxes = [bbox for _, bbox in annotations]
                obj_id_list = [oid for oid, _ in annotations]

                # Test points against bboxes
                obj_ids_per_point, _ = points_in_rotated_3d_bbox(points, bboxes)

                # Mark points inside moving bboxes as dynamic
                dynamic_mask = obj_ids_per_point >= 0
                is_dynamic_frame[dynamic_mask] = 1.0

                # Assign init velocity from tracked object velocity
                if getattr(args, 'od_velocity_init', True):
                    for local_idx, obj_id in enumerate(obj_id_list):
                        if obj_id in obj_velocities:
                            point_mask = obj_ids_per_point == local_idx
                            init_velocity_frame[point_mask] = obj_velocities[obj_id]
        # global_lidar_points = (points_homo @ lidar2global.T)[:, :3]
        points_is_dynamic.append(is_dynamic_frame)
        points_obj_velocity.append(init_velocity_frame)

        # Timestamp for dynamic scene: linearly map frame index to time_duration
        t = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / max(frames - 1, 1)
        point_time = np.full_like(points[:, :1], t)
        points_time.append(point_time)

        idx = frame_idx

        # World-to-lidar (camera) transform
        # KITTI-360 convention: R is stored transposed (for glm in CUDA)
        w2l = np.array([0, -1, 0, 0,
                        0, 0, -1, 0,
                        1, 0, 0, 0,
                        0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(LIDAR2EGO) @ np.linalg.inv(lidar2global)
        R = np.transpose(w2l[:3, :3])
        T = w2l[:3, 3]
        points_cam = points @ R + T

        # Forward half (front 180 degrees)
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T,
                                    timestamp=t, pointcloud_camera=points_cam, intensity=intensity,
                                    towards='forward'))

        # Backward half (back 180 degrees)
        R_back = R @ np.array([-1, 0, 0,
                               0, 1, 0,
                               0, 0, -1]).reshape(3, 3)
        T_back = T * np.array([-1, 1, -1])
        points_cam_back = points @ R_back + T_back
        cam_infos.append(CameraInfo(uid=idx + frames, R=R_back, T=T_back,
                                    timestamp=t, pointcloud_camera=points_cam_back, intensity=intensity,
                                    towards='backward'))

        if debug_cuda and frame_idx >= 15:
            break

    if len(point_list) == 0:
        raise RuntimeError("No point cloud files found! Please check that raw .bin files exist.")

    pointcloud = np.concatenate(point_list, axis=0)
    assert pointcloud.shape[0] > 0, \
        "Concatenated point cloud is empty — all .bin files may have been skipped"
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    pointcloud_is_dynamic = np.concatenate(points_is_dynamic, axis=0)
    pointcloud_init_velocity = np.concatenate(points_obj_velocity, axis=0)

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

    if not args.test_only:
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, args.dynamic)
        np.savez(os.path.join(args.model_path, 'transform_poses_pca.npz'),
                 transform=transform, scale_factor=scale_factor)
        c2ws = pad_poses(c2ws)
    else:
        data_npz = np.load(os.path.join(args.model_path, 'transform_poses_pca.npz'))
        transform = data_npz['transform']
        scale_factor = data_npz['scale_factor'].item()
        c2ws = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2ws)
        c2ws[:, :3, 3] *= scale_factor

    for idx, cam_info in enumerate(cam_infos):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor

    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    # Transform init_velocity to PCA-aligned coordinate system (rotation only)
    if pointcloud_init_velocity is not None and np.any(pointcloud_init_velocity != 0):
        pointcloud_init_velocity = pointcloud_init_velocity @ transform[:3, :3].T
    args.scale_factor = float(scale_factor)

    mod = args.cam_num  # 2 for forward/backward

    # Validate forward/backward camera pairing integrity
    assert len(cam_infos) % mod == 0, \
        f"cam_infos length {len(cam_infos)} not divisible by cam_num={mod}, forward/backward pairing broken"

    # val_frame_ids stores local frame indices
    if eval_mode:
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
