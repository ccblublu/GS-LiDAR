#!/usr/bin/env python3
"""Verify PCI IMU-frame loading: stacked world PCD + refine-style GT exports.

Fixes that this script checks:
  - points + poses are IMU-frame → panoramic RV uses lidar2ego = I
  - supervision GT is built from the corrected camera frame

Outputs (under model_path/verify_imu_frame/ by default):
  first100_world.pcd          — first 100 training frames fused in world
  frame_XXXX.png              — GT depth (top) | GT depth (all) | intensity
  frame_XXXX_gt.ply           — all-source supervision cloud (RV local, like refine)
  frame_XXXX_input.ply        — top-only input cloud (RV local)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scene.gacrt_loader import (
    BODY_TO_RV,
    filter_lidar_sources,
    get_lidar2ego,
    load_lidar_xyz_intensity,
)
from utils.camera_utils import project_points_to_rv
from utils.general_utils import visualize_depth


def write_pcd_xyz_intensity_source(path: Path, xyz, intensity, source=None):
    n = int(xyz.shape[0])
    if source is None:
        source = np.zeros(n, dtype=np.int32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity source\n"
        "SIZE 4 4 4 4 4\n"
        "TYPE F F F F I\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    rec = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("source", "<i4"),
        ],
    )
    rec["x"] = xyz[:, 0].astype(np.float32)
    rec["y"] = xyz[:, 1].astype(np.float32)
    rec["z"] = xyz[:, 2].astype(np.float32)
    rec["intensity"] = intensity.astype(np.float32)
    rec["source"] = source.astype(np.int32)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


def write_ply_xyz_intensity(path: Path, xyz, intensity):
    """Match train.py refine export: xyz + turbo colormap from intensity."""
    from matplotlib import cm
    from plyfile import PlyData, PlyElement

    inten = np.asarray(intensity, dtype=np.float64).reshape(-1)
    inten = np.clip(inten, 0.0, 1.0)
    colors = (cm.get_cmap("turbo")(inten)[:, :3] * 255).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(len(xyz), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def stitch_half_panos(front, back):
    """front/back: (1,H,W) → full 360 (1,H,2W) matching render_range_map layout."""
    _, h, w = front.shape
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    out = np.zeros((1, h, w * 2), dtype=np.float32)
    out[:, :, breaks[1]:breaks[2]] = front
    out[:, :, breaks[2]:breaks[3]] = back[:, :, 0:(breaks[3] - breaks[2])]
    out[:, :, breaks[0]:breaks[1]] = back[:, :, (w - breaks[1] + breaks[0]):w]
    return out


def pano_to_xyz(depth_pano, vfov, hfov):
    """Numpy port of utils.graphics_utils.pano_to_lidar for 360° depth."""
    depth = depth_pano[0]
    h, w = depth.shape
    mask = depth > 0
    theta, phi = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    v_range = vfov[1] - vfov[0]
    h_range = hfov[1] - hfov[0]
    # Full 360 stitch uses hfov spanning [-180, 180] effectively via 2*half width.
    # depth_pano width = 2 * half_w, so treat as hfov=[-180,180].
    theta = (90 - vfov[1] + theta / h * v_range) * np.pi / 180
    phi = (-180.0 + phi / w * 360.0) * np.pi / 180
    dx = np.sin(theta) * np.sin(phi)
    dz = np.sin(theta) * np.cos(phi)
    dy = -np.cos(theta)
    dirs = np.stack([dx, dy, dz], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True).clip(1e-8)
    pts = dirs[mask] * depth[mask][:, None]
    return pts.astype(np.float32), mask


def load_frame_cloud(full_path, args):
    xyz, intensity, source = load_lidar_xyz_intensity(str(full_path), return_source=True)
    intensity = intensity / 255.0
    xyz, intensity, input_mask, n_body, source = filter_lidar_sources(xyz, intensity, source, args)
    return xyz, intensity, input_mask, n_body, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gacrt_nvs_PCI_20260825.yaml")
    parser.add_argument("--source_path", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--n_stack", type=int, default=100)
    parser.add_argument("--n_preview", type=int, default=10)
    parser.add_argument("--max_stack_pts", type=int, default=8_000_000)
    args_cli = parser.parse_args()

    base = OmegaConf.load(_ROOT / "configs/base.yaml")
    scene = OmegaConf.load(_ROOT / args_cli.config)
    conf = OmegaConf.merge(base, scene)
    if args_cli.source_path:
        conf.source_path = args_cli.source_path

    out_dir = Path(args_cli.out_dir) if args_cli.out_dir else (
        Path(conf.model_path) / "verify_imu_frame"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(conf.source_path) / conf.sequence_id / f"transforms_{conf.sequence_id}_all.json"
    json_data = json.loads(json_path.read_text())
    frames = json_data["frames"]
    lidar2ego = get_lidar2ego(conf, json_data)
    print(f"lidar2ego is identity: {np.allclose(lidar2ego, np.eye(4))}")
    print(f"pose_frame={json_data.get('pose_frame')} point_frame={json_data.get('point_frame')}")
    if not np.allclose(lidar2ego, np.eye(4)):
        raise SystemExit("Expected identity lidar2ego for IMU-frame PCI data")

    vfov = list(conf.vfov)
    hfov = list(conf.hfov)
    hw = list(conf.hw)
    n_stack = min(args_cli.n_stack, len(frames))
    n_preview = min(args_cli.n_preview, len(frames))

    # --- first N stacked in world ---
    xs, ys, zs, ints, srcs = [], [], [], [], []
    n_body_total = 0
    for i in range(n_stack):
        fr = frames[i]
        path = Path(conf.source_path) / fr["lidar_path"]
        xyz, intensity, input_mask, n_body, _ = load_frame_cloud(path, conf)
        n_body_total += n_body
        T = np.asarray(fr["lidar2world"], dtype=np.float64)
        homo = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
        world = (homo @ T.T)[:, :3]
        # mark source: 0 top input, 1 fill-in kept for supervision
        src_out = np.where(input_mask, 0, 1).astype(np.int32)
        xs.append(world[:, 0].astype(np.float32))
        ys.append(world[:, 1].astype(np.float32))
        zs.append(world[:, 2].astype(np.float32))
        ints.append(intensity.astype(np.float32) * 255.0)
        srcs.append(src_out)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"stack {i + 1}/{n_stack} n={len(world)} body_rm={n_body}")

    xyz_w = np.stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)], axis=1)
    inten_w = np.concatenate(ints)
    src_w = np.concatenate(srcs)
    if len(xyz_w) > args_cli.max_stack_pts:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz_w), args_cli.max_stack_pts, replace=False)
        xyz_w, inten_w, src_w = xyz_w[sel], inten_w[sel], src_w[sel]
    stack_path = out_dir / f"first{n_stack}_world.pcd"
    write_pcd_xyz_intensity_source(stack_path, xyz_w, inten_w, src_w)
    print(f"wrote {stack_path} n={len(xyz_w)} body_removed_total={n_body_total}")

    # --- first K frames: refine-style GT export ---
    for i in range(n_preview):
        fr = frames[i]
        path = Path(conf.source_path) / fr["lidar_path"]
        xyz, intensity, input_mask, n_body, _ = load_frame_cloud(path, conf)
        T_w = np.asarray(fr["lidar2world"], dtype=np.float64)
        homo = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
        world = (homo @ T_w.T)[:, :3]
        world_in = world[input_mask]
        inten_in = intensity[input_mask]

        w2l = BODY_TO_RV @ np.linalg.inv(lidar2ego) @ np.linalg.inv(T_w)
        R = np.transpose(w2l[:3, :3])
        T = w2l[:3, 3]
        cam = world @ R + T
        cam_in = world_in @ R + T

        R_back = R @ np.array([-1, 0, 0, 0, 1, 0, 0, 0, -1], float).reshape(3, 3)
        T_back = T * np.array([-1, 1, -1])
        cam_b = world @ R_back + T_back
        cam_b_in = world_in @ R_back + T_back

        d_f, i_f = project_points_to_rv(cam, intensity, hw[0], hw[1], vfov, hfov)
        d_b, i_b = project_points_to_rv(cam_b, intensity, hw[0], hw[1], vfov, hfov)
        d_fi, _ = project_points_to_rv(cam_in, inten_in, hw[0], hw[1], vfov, hfov)
        d_bi, _ = project_points_to_rv(cam_b_in, inten_in, hw[0], hw[1], vfov, hfov)

        depth_all = stitch_half_panos(d_f.astype(np.float32), d_b.astype(np.float32))
        inten_all = stitch_half_panos(i_f.astype(np.float32), i_b.astype(np.float32))
        depth_in = stitch_half_panos(d_fi.astype(np.float32), d_bi.astype(np.float32))

        # PNG layout similar to refine: GT depth | input(top) depth | intensity
        t_depth = torch.from_numpy(depth_all)
        t_in = torch.from_numpy(depth_in)
        t_int = torch.from_numpy(inten_all).clamp(0, 1).repeat(3, 1, 1).cuda()
        grid = make_grid(
            [visualize_depth(t_depth, scale_factor=1.0),
             visualize_depth(t_in, scale_factor=1.0),
             t_int],
            nrow=1, padding=0,
        )
        png_path = out_dir / f"{i:03d}.png"
        save_image(grid, str(png_path))

        pts_gt, mask_gt = pano_to_xyz(depth_all, vfov, hfov)
        inten_gt = inten_all[0][mask_gt]
        write_ply_xyz_intensity(out_dir / f"{i:03d}_gt.ply", pts_gt, inten_gt)

        pts_in, mask_in = pano_to_xyz(depth_in, vfov, hfov)
        inten_pi = inten_all[0][mask_in] if mask_in.shape == mask_gt.shape else np.ones(mask_in.sum())
        # intensity from input pano
        inten_in_pano = stitch_half_panos(
            project_points_to_rv(cam_in, inten_in, hw[0], hw[1], vfov, hfov)[1].astype(np.float32),
            project_points_to_rv(cam_b_in, inten_in, hw[0], hw[1], vfov, hfov)[1].astype(np.float32),
        )
        inten_pi = inten_in_pano[0][mask_in]
        write_ply_xyz_intensity(out_dir / f"{i:03d}_input.ply", pts_in, inten_pi)

        cov_all = float((depth_all > 0).mean())
        cov_in = float((depth_in > 0).mean())
        r_gt = np.linalg.norm(pts_gt, axis=1) if len(pts_gt) else np.array([0.0])
        print(
            f"preview {i:03d}: body={n_body} top={int(input_mask.sum())} "
            f"sup={len(xyz)} cov_all={cov_all:.1%} cov_top={cov_in:.1%} "
            f"gt_r_med={float(np.median(r_gt)):.2f}m"
        )

    meta = {
        "lidar2ego_identity": True,
        "pose_frame": json_data.get("pose_frame"),
        "point_frame": json_data.get("point_frame"),
        "n_stack": n_stack,
        "n_preview": n_preview,
        "stack_pcd": str(stack_path),
        "out_dir": str(out_dir),
    }
    (out_dir / "verify_meta.json").write_text(json.dumps(meta, indent=2))
    print("done →", out_dir)


if __name__ == "__main__":
    main()
