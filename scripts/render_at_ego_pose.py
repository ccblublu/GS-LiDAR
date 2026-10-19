#!/usr/bin/env python3
#
# Render GS-LiDAR from a modified ego / LiDAR pose (inference-only NVS).
#
# Examples:
#   # Shift ego 2m forward (GACRT ego: x=forward, y=left, z=up)
#   python scripts/render_at_ego_pose.py \
#       --config configs/gacrt_nvs_GACRT014_1740908194.yaml \
#       source_path=data \
#       model_path=eval_output/gacrt_reconstruction/GACRT014_1740908194 \
#       --ego_offset 2.0 0.0 0.0 --split test --max_frames 5
#
#   # Shift + yaw, compare with original pose
#   python scripts/render_at_ego_pose.py \
#       --config configs/gacrt_nvs_GACRT014_1740908194.yaml \
#       source_path=data \
#       model_path=eval_output/gacrt_reconstruction/GACRT014_1740908194 \
#       --ego_offset 1.0 0.5 0.0 --ego_yaw_deg 5 --save_comparison
#
# Outputs per frame × sensor (QT128 blind: left/right/front/back):
#   {frame}_{sensor}_render.png  - novel pose render
#   {frame}_{sensor}_gt.png      - GT intensity panorama
#   {frame}_{sensor}.ply         - novel view in that sensor's GT RV frame
#   {frame}_{sensor}_gt.ply      - GT point cloud (same sensor RV frame)
#   {frame}_{sensor}_compare.png - optional, with --save_comparison
#
# Multi-sensor also writes one fused cloud per frame in IMU / car frame
# (x=forward, y=left, z=up, meters):
#   {frame}_imu.ply / {frame}_imu_gt.ply
#
# Novel per-sensor PLY is transformed from the offset camera back to the
# original GT origin so both clouds share one frame for overlay comparison.
#
#   # Use another frame's pose but keep current frame timestamp (dynamic content)
#   python scripts/render_at_ego_pose.py \
#       --config configs/gacrt_nvs_GACRT014_1740908194.yaml \
#       source_path=data \
#       model_path=eval_output/gacrt_reconstruction/GACRT014_1740908194 \
#       --frame_idx 10 --pose_from_frame 50
#
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scene.gacrt_loader import BODY_TO_RV, HESAI_TO_BODY, load_blind_extrinsics
from utils.graphics_utils import getWorld2View2
from utils.general_utils import seed_everything
from utils.system_utils import save_ply

from render_without_object import (
    apply_raydrop_mask,
    camera_pair_for_frame,
    iter_frame_sensor_pairs,
    list_sensor_ids,
    load_merged_config,
    load_model,
    load_transforms_frames,
    render_range_map_masked,
    select_cameras,
    sensor_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render GS-LiDAR from a modified ego pose (inference only)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--frame_idx", type=int, action="append", default=None,
                        help="Only render these frame indices (repeatable)")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--ego_offset", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                        default=[0.0, 0.0, 0.0],
                        help="Ego-frame translation in meters (x=forward, y=left, z=up)")
    parser.add_argument("--ego_yaw_deg", type=float, default=0.0,
                        help="Ego-frame yaw offset in degrees (rotation about z)")
    parser.add_argument("--pose_from_frame", type=int, default=None,
                        help="Borrow ego pose from this frame index (keep current timestamp)")
    parser.add_argument("--no_raydrop_prior", action="store_true")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save original-pose vs novel-pose comparison image")
    parser.add_argument("--no_ply", action="store_true")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument(
        "--sensor_id", type=int, action="append", default=None,
        help="Only render these blind-lidar slots (0=left,1=right,2=front,3=back). "
             "Default: all sensors present in the scene.",
    )
    parser.add_argument(
        "--offset_set", type=str, default=None, choices=["gacrt014"],
        help="Render a named batch of ego offsets after one model load",
    )
    known_args, cli_overrides = parser.parse_known_args()
    return known_args, cli_overrides


# Same offset grid previously used for GACRT014 render_at_ego_pose.
OFFSET_SET_GACRT014 = [
    (0.0, 0.0, 3.0, 0.0),
    (0.0, 0.0, 5.0, 0.0),
    (0.3, 0.0, 0.0, 1.0),
    (0.0, 3.0, 0.0, 0.0),
    (0.0, 5.0, 0.0, 0.0),
    (1.0, 0.5, 0.0, 5.0),
    (2.0, 1.5, 0.0, 0.0),
    (3.0, 0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0, 1.0),
    (5.0, 0.0, 0.0, 0.0),
]


def build_ego_delta_4x4(dx, dy, dz, yaw_deg, scale_factor):
    """4x4 delta in ego / LiDAR frame (applied as c2w_new = c2w @ delta)."""
    t = np.array([dx, dy, dz], dtype=np.float64) * float(scale_factor)
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rot
    delta[:3, 3] = t
    return delta


def c2w_from_camera(camera):
    w2v = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    return np.linalg.inv(w2v)


def update_camera_from_c2w(camera, c2w):
    w2c = np.linalg.inv(c2w)
    camera.R = np.transpose(w2c[:3, :3])
    camera.T = w2c[:3, 3].copy()
    trans = np.asarray(camera.trans, dtype=np.float64)
    scale = float(camera.scale)
    camera.world_view_transform = torch.tensor(
        getWorld2View2(camera.R, camera.T, trans, scale), dtype=torch.float32,
    ).transpose(0, 1).cuda()
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    camera.c2w = camera.world_view_transform.transpose(0, 1).inverse()


def snapshot_camera_poses(cameras):
    snapshots = {}
    for cam in cameras:
        snapshots[cam.uid] = {
            "R": np.array(cam.R, copy=True),
            "T": np.array(cam.T, copy=True),
        }
    return snapshots


def restore_camera_pose(camera, snapshot):
    update_camera_from_c2w(camera, c2w_from_rt(snapshot["R"], snapshot["T"]))


def c2w_from_rt(R, T):
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.transpose(R)
    w2c[:3, 3] = T
    return np.linalg.inv(w2c)


def apply_rig_delta(cam_front, cam_back, delta_ego):
    """Apply ego-local delta to the whole LiDAR rig coherently.

    We first convert the ego-local delta into a world-frame transform using the
    front camera as the rig anchor, then left-multiply both front/back camera
    poses. This preserves their rigid relation. ``cam_back`` may be ``None``
    for half-only (cam_num=1) scenes.
    """
    c2w_front = c2w_from_camera(cam_front)
    world_delta = c2w_front @ delta_ego @ np.linalg.inv(c2w_front)

    cams = (cam_front,) if cam_back is None else (cam_front, cam_back)
    for cam in cams:
        c2w = c2w_from_camera(cam)
        update_camera_from_c2w(cam, world_delta @ c2w)


def apply_pose_from_reference(cam_front, cam_back, ref_front, ref_back, delta_ego):
    # Start from reference rig pose, then apply a coherent rig delta.
    update_camera_from_c2w(cam_front, c2w_from_camera(ref_front))
    if cam_back is not None and ref_back is not None:
        update_camera_from_c2w(cam_back, c2w_from_camera(ref_back))
    apply_rig_delta(cam_front, cam_back, delta_ego)


def intensity_to_rgb(intensity):
    """Map intensity [1,H,W] to RGB visualization."""
    return intensity.clamp(0, 1).repeat(3, 1, 1)


def pano_to_lidar_with_intensity(depth, intensity, vfov, hfov):
    """Convert panoramic depth + intensity maps to colored point cloud."""
    mask = depth > 0
    panorama_height, panorama_width = depth.shape[-2:]
    theta, phi = torch.meshgrid(
        torch.arange(panorama_height, device=depth.device),
        torch.arange(panorama_width, device=depth.device),
        indexing="ij",
    )

    vertical_degree_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

    horizontal_degree_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = torch.nn.functional.normalize(directions, dim=0)

    points_xyz = (directions * depth)[:, mask[0]].permute(1, 0)
    intensity_vals = intensity[:, mask[0]].permute(1, 0)
    return points_xyz, intensity_vals


def stitch_gt_range_map(cam_front, cam_back, hw):
    """Stitch GT depth/intensity panoramas from camera maps (no Gaussian render).

    ``cam_back is None`` returns the single forward half at HxW.
    """
    if cam_back is None:
        return cam_front.pts_depth.cuda(), cam_front.pts_intensity.cuda()

    h, w = hw
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    gt_depth_pano = torch.zeros([1, h, w * 2], device="cuda")
    gt_intensity_pano = torch.zeros([1, h, w * 2], device="cuda")

    for idx, viewpoint in enumerate([cam_front, cam_back]):
        depth_gt = viewpoint.pts_depth.cuda()
        intensity_gt = viewpoint.pts_intensity.cuda()
        if idx % 2 == 0:
            gt_depth_pano[:, :, breaks[1]:breaks[2]] = depth_gt
            gt_intensity_pano[:, :, breaks[1]:breaks[2]] = intensity_gt
            continue

        gt_depth_pano[:, :, breaks[2]:breaks[3]] = depth_gt[:, :, 0:(breaks[3] - breaks[2])]
        gt_depth_pano[:, :, breaks[0]:breaks[1]] = depth_gt[:, :, (w - breaks[1] + breaks[0]):w]
        gt_intensity_pano[:, :, breaks[2]:breaks[3]] = intensity_gt[:, :, 0:(breaks[3] - breaks[2])]
        gt_intensity_pano[:, :, breaks[0]:breaks[1]] = intensity_gt[:, :, (w - breaks[1] + breaks[0]):w]

    return gt_depth_pano, gt_intensity_pano


def transform_points(points, T):
    """Apply a 4x4 transform to (N, 3) points. T maps src-local → dst-local."""
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    homo = torch.cat([points, ones], dim=1)
    T_t = torch.as_tensor(T, device=points.device, dtype=points.dtype)
    return (T_t @ homo.T).T[:, :3]


# RV (GS panoramic) → Hesai housing → IMU. Pure geometry; translations in meters.
_RV_TO_HOUSING = np.linalg.inv(HESAI_TO_BODY) @ np.linalg.inv(BODY_TO_RV)


def imu_from_rv_matrix(imu_from_lidar):
    """4x4: scaled-RV meters → IMU (x-forward, y-left, z-up)."""
    return np.asarray(imu_from_lidar, dtype=np.float64) @ _RV_TO_HOUSING


def depth_to_points_intensity(depth, intensity, vfov, T_out_from_local=None, hfov=(-180, 180)):
    """Decode pano → (N,3) points + (N,1) intensity in the (optionally remapped) RV frame."""
    points, inten = pano_to_lidar_with_intensity(depth, intensity, vfov, hfov)
    if T_out_from_local is not None:
        points = transform_points(points, T_out_from_local)
    return points, inten


def save_intensity_ply(depth, intensity, vfov, scale_factor, path, T_out_from_local=None,
                       hfov=(-180, 180)):
    points, inten = depth_to_points_intensity(
        depth, intensity, vfov, T_out_from_local, hfov=hfov,
    )
    inten_np = inten.clamp(0, 1).detach().cpu().numpy()
    rgbs = np.repeat(inten_np, 3, axis=1)
    save_ply(points / scale_factor, path, rgbs=rgbs)
    return points, inten


def points_rv_to_imu(points_rv, imu_from_lidar, scale_factor):
    """RV (scene-scaled) → IMU meters via blind-lidar extrinsic."""
    pts_m = points_rv / float(scale_factor)
    return transform_points(pts_m, imu_from_rv_matrix(imu_from_lidar))


def append_imu_cloud(bucket, frame_idx, points_imu, intensity):
    if points_imu is None or points_imu.shape[0] == 0:
        return
    bucket[frame_idx]["xyz"].append(points_imu.detach().cpu().numpy())
    inten_np = intensity.clamp(0, 1).detach().cpu().numpy()
    bucket[frame_idx]["rgb"].append(np.repeat(inten_np, 3, axis=1))


def flush_imu_clouds(bucket, output_dir, suffix):
    """Write accumulated per-frame IMU clouds: `{frame:03d}_imu{suffix}.ply`."""
    for frame_idx in sorted(bucket.keys()):
        chunks_xyz = bucket[frame_idx]["xyz"]
        chunks_rgb = bucket[frame_idx]["rgb"]
        if not chunks_xyz:
            continue
        xyz = np.concatenate(chunks_xyz, axis=0)
        rgb = np.concatenate(chunks_rgb, axis=0)
        path = os.path.join(output_dir, f"{frame_idx:03d}_imu{suffix}.ply")
        save_ply(xyz, path, rgbs=rgb)


def default_output_dir(model_path, args_read):
    dx, dy, dz = args_read.ego_offset
    tag = f"offset_{dx:g}_{dy:g}_{dz:g}_yaw{args_read.ego_yaw_deg:g}"
    if args_read.pose_from_frame is not None:
        tag += f"_pose{args_read.pose_from_frame}"
    return os.path.join(model_path, "render_at_ego_pose", tag)


def offset_jobs(args_read):
    if args_read.offset_set == "gacrt014":
        return [
            {"ego_offset": [dx, dy, dz], "ego_yaw_deg": yaw}
            for dx, dy, dz, yaw in OFFSET_SET_GACRT014
        ]
    # has_delta = (
    #     any(abs(v) > 1e-9 for v in args_read.ego_offset)
    #     or abs(args_read.ego_yaw_deg) > 1e-9
    #     or args_read.pose_from_frame is not None
    # )
    # if not has_delta:
    #     raise SystemExit(
    #         "Specify --offset_set, or a pose change via --ego_offset / "
    #         "--ego_yaw_deg / --pose_from_frame."
        # )
    return [{
        "ego_offset": list(args_read.ego_offset),
        "ego_yaw_deg": args_read.ego_yaw_deg,
    }]


def main():
    args_read, cli_overrides = parse_args()
    conf = load_merged_config(args_read, cli_overrides)
    seed_everything(conf.seed)
    jobs = offset_jobs(args_read)

    scene, gaussians, env_map, iteration = load_model(
        conf, args_read.iteration, use_raydrop=not args_read.no_raydrop_prior,
    )

    poses, val_frame_ids = load_transforms_frames(conf)
    n_frames = len(poses)
    cameras, frame_ids = select_cameras(scene, conf, args_read.split, val_frame_ids)

    if args_read.frame_idx:
        wanted = set(args_read.frame_idx)
        frame_ids = [fid for fid in frame_ids if fid in wanted]
        if not frame_ids:
            raise SystemExit(f"No frames matched --frame_idx {sorted(wanted)}")

    if args_read.max_frames is not None:
        frame_ids = frame_ids[: args_read.max_frames]

    if args_read.pose_from_frame is not None:
        if not (0 <= args_read.pose_from_frame < n_frames):
            raise SystemExit(f"--pose_from_frame must be in [0, {n_frames})")

    pose_snapshots = snapshot_camera_poses(cameras)

    if args_read.offset_set and args_read.output_dir:
        raise SystemExit("--output_dir cannot be combined with --offset_set")

    for job_i, job in enumerate(jobs):
        args_read.ego_offset = job["ego_offset"]
        args_read.ego_yaw_deg = job["ego_yaw_deg"]
        print(
            f"[{job_i + 1}/{len(jobs)}] offset={args_read.ego_offset} "
            f"yaw={args_read.ego_yaw_deg:g}"
        )
        render_offset_job(
            args_read, conf, cameras, frame_ids, n_frames, pose_snapshots,
            gaussians, env_map, iteration,
        )


def render_offset_job(
    args_read, conf, cameras, frame_ids, n_frames, pose_snapshots,
    gaussians, env_map, iteration,
):
    delta_ego = build_ego_delta_4x4(
        *args_read.ego_offset, args_read.ego_yaw_deg, conf.scale_factor,
    )

    output_dir = args_read.output_dir or default_output_dir(conf.model_path, args_read)
    os.makedirs(output_dir, exist_ok=True)

    scale = conf.resolution_scales[-1]
    h, w = conf.hw[0] // scale, conf.hw[1] // scale
    bg_color = [1, 1, 1, 1] if conf.white_background else [0, 0, 0, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    sensor_ids = (
        sorted(set(args_read.sensor_id)) if args_read.sensor_id
        else list_sensor_ids(cameras)
    )
    multi_sensor = len(sensor_ids) > 1
    half_only = int(getattr(conf, "cam_num", 2) or 2) == 1
    ply_hfov = tuple(conf.hfov) if half_only else (-180, 180)

    # Blind-lidar → IMU extrinsics (source_id 1..4). Slot = source_id - 1.
    blind_extrinsics = None
    if multi_sensor and getattr(conf, "blind_lidar_origins", False):
        calib_dir = getattr(conf, "calib_dir", None)
        if not calib_dir:
            raise SystemExit("blind_lidar_origins requires calib_dir to fuse into IMU")
        blind_extrinsics = load_blind_extrinsics(calib_dir)

    meta = {
        "iteration": iteration,
        "split": args_read.split,
        "ego_offset_m": list(args_read.ego_offset),
        "ego_yaw_deg": args_read.ego_yaw_deg,
        "pose_from_frame": args_read.pose_from_frame,
        "scale_factor": conf.scale_factor,
        "ply_frame": "original_gt_rv",
        "ply_hfov": list(ply_hfov),
        "half_only": half_only,
        "imu_fused": bool(blind_extrinsics),
        "imu_frame": "x-forward y-left z-up meters" if blind_extrinsics else None,
        "sensor_ids": sensor_ids,
        "frames": [],
    }

    pairs = list(iter_frame_sensor_pairs(cameras, frame_ids, n_frames, sensor_ids))
    if not pairs:
        raise SystemExit(
            f"No camera pairs for frame_ids={frame_ids} sensor_ids={sensor_ids}"
        )

    imu_render = defaultdict(lambda: {"xyz": [], "rgb": []})
    imu_gt = defaultdict(lambda: {"xyz": [], "rgb": []})

    with torch.no_grad():
        for frame_idx, sid, cam_front, cam_back in tqdm(pairs, desc="Render at ego pose"):
            timestamp = cam_front.timestamp
            sname = sensor_name(sid)
            # Single-sensor scenes keep legacy `{frame}` names; multi-sensor
            # QT128 writes `{frame}_{left|right|front|back}`.
            frame_tag = (
                f"{frame_idx:03d}_{sname}" if multi_sensor else f"{frame_idx:03d}"
            )

            restore_camera_pose(cam_front, pose_snapshots[cam_front.uid])
            if cam_back is not None:
                restore_camera_pose(cam_back, pose_snapshots[cam_back.uid])
            c2w_gt = c2w_from_camera(cam_front)

            gt_depth, gt_inten = stitch_gt_range_map(cam_front, cam_back, [h, w])

            inten_orig = raydrop_orig = None
            if args_read.save_comparison:
                depth_orig, inten_orig, raydrop_orig, _, _ = render_range_map_masked(
                    conf, cam_front, cam_back, gaussians, background, env_map, [h, w],
                )

            if args_read.pose_from_frame is not None:
                ref_front, ref_back = camera_pair_for_frame(
                    cameras, args_read.pose_from_frame, n_frames, sensor_id=sid,
                )
                apply_pose_from_reference(cam_front, cam_back, ref_front, ref_back, delta_ego)
            else:
                apply_rig_delta(cam_front, cam_back, delta_ego)

            cam_front.timestamp = timestamp
            if cam_back is not None:
                cam_back.timestamp = timestamp
            c2w_novel = c2w_from_camera(cam_front)
            # Offset camera → original GT sensor frame (same origin as _gt.ply).
            T_gt_from_novel = np.linalg.inv(c2w_gt) @ c2w_novel

            depth_novel, inten_novel, raydrop_novel, _, _ = render_range_map_masked(
                conf, cam_front, cam_back, gaussians, background, env_map, [h, w],
            )

            restore_camera_pose(cam_front, pose_snapshots[cam_front.uid])
            if cam_back is not None:
                restore_camera_pose(cam_back, pose_snapshots[cam_back.uid])

            depth_novel, inten_novel, _ = apply_raydrop_mask(
                depth_novel[[0]], inten_novel, raydrop_novel,
            )
            if args_read.save_comparison:
                _, inten_orig, _ = apply_raydrop_mask(
                    depth_orig[[0]], inten_orig, raydrop_orig,
                )

            if not args_read.no_ply:
                pts_rv, inten_pts = save_intensity_ply(
                    depth_novel, inten_novel, conf.vfov, conf.scale_factor,
                    os.path.join(output_dir, f"{frame_tag}.ply"),
                    T_out_from_local=T_gt_from_novel,
                    hfov=ply_hfov,
                )
                pts_gt_rv, inten_gt_pts = save_intensity_ply(
                    gt_depth, gt_inten, conf.vfov, conf.scale_factor,
                    os.path.join(output_dir, f"{frame_tag}_gt.ply"),
                    hfov=ply_hfov,
                )
                if blind_extrinsics is not None:
                    source_id = int(sid) + 1
                    if source_id not in blind_extrinsics:
                        raise SystemExit(
                            f"Missing blind extrinsic for sensor slot {sid} "
                            f"(source_id={source_id})"
                        )
                    T_lidar = blind_extrinsics[source_id]["imu_from_lidar"]
                    append_imu_cloud(
                        imu_render, frame_idx,
                        points_rv_to_imu(pts_rv, T_lidar, conf.scale_factor),
                        inten_pts,
                    )
                    append_imu_cloud(
                        imu_gt, frame_idx,
                        points_rv_to_imu(pts_gt_rv, T_lidar, conf.scale_factor),
                        inten_gt_pts,
                    )

            save_image(
                intensity_to_rgb(inten_novel),
                os.path.join(output_dir, f"{frame_tag}_render.png"),
            )
            save_image(
                intensity_to_rgb(gt_inten),
                os.path.join(output_dir, f"{frame_tag}_gt.png"),
            )

            if args_read.save_comparison:
                grid = make_grid(
                    [intensity_to_rgb(inten_orig), intensity_to_rgb(inten_novel),
                     intensity_to_rgb(gt_inten), intensity_to_rgb(inten_novel)],
                    nrow=2,
                )
                save_image(grid, os.path.join(output_dir, f"{frame_tag}_compare.png"))

            meta["frames"].append({
                "frame_idx": frame_idx,
                "sensor_id": sid,
                "sensor": sname,
                "timestamp": float(timestamp),
                "camera_center": c2w_novel[:3, 3].tolist(),
                "gt_camera_center": c2w_gt[:3, 3].tolist(),
                "pose_from_frame": args_read.pose_from_frame,
            })

    if not args_read.no_ply and blind_extrinsics is not None:
        flush_imu_clouds(imu_render, output_dir, "")
        flush_imu_clouds(imu_gt, output_dir, "_gt")
        meta["imu_frames"] = sorted(imu_render.keys())

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    n_imu = len(imu_render) if blind_extrinsics is not None else 0
    print(
        f"Done. {len(pairs)} renders "
        f"({len(frame_ids)} frames × {len(sensor_ids)} sensors)"
        + (f", {n_imu} IMU-fused frames" if n_imu else "")
        + f" → {output_dir}"
    )


if __name__ == "__main__":
    main()
