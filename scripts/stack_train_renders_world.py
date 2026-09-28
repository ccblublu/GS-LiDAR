#!/usr/bin/env python3
"""Stack training-rendered LiDAR frames into one world-frame PCD + BEV.

Reads ray_drop_datasets/render_{split}/*.pt (raydrop, intensity, depth),
converts panoramic depth to points, undoes scale_factor, maps RV → IMU → world
via transforms JSON poses, writes a PCD and a BEV image with ego trajectory.

Example:
  python scripts/stack_train_renders_world.py \
    --config configs/gacrt_nvs_PCI_20260825.yaml \
    --n_frames 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scene.gacrt_loader import BODY_TO_RV


def pano_to_xyz(depth: np.ndarray, vfov, hfov=(-180.0, 180.0)):
    """depth: (1,H,W) or (H,W) → (N,3) in RV camera frame."""
    if depth.ndim == 3:
        depth = depth[0]
    h, w = depth.shape
    mask = depth > 0
    theta, phi = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    v_range = vfov[1] - vfov[0]
    h_range = hfov[1] - hfov[0]
    theta = (90.0 - vfov[1] + theta / h * v_range) * np.pi / 180.0
    phi = (hfov[0] + phi / w * h_range) * np.pi / 180.0
    dx = np.sin(theta) * np.sin(phi)
    dz = np.sin(theta) * np.cos(phi)
    dy = -np.cos(theta)
    dirs = np.stack([dx, dy, dz], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True).clip(1e-8)
    pts = dirs[mask] * depth[mask][:, None]
    return pts.astype(np.float64), mask


def write_pcd(path: Path, xyz: np.ndarray, intensity: np.ndarray | None = None):
    n = int(xyz.shape[0])
    if intensity is None:
        intensity = np.ones(n, dtype=np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    rec["x"] = xyz[:, 0].astype(np.float32)
    rec["y"] = xyz[:, 1].astype(np.float32)
    rec["z"] = xyz[:, 2].astype(np.float32)
    rec["intensity"] = intensity.astype(np.float32)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


def rv_to_world(points_rv: np.ndarray, imu2world: np.ndarray) -> np.ndarray:
    """RV (GS panoramic) local → IMU body → world. lidar2ego assumed identity."""
    rv2imu = np.linalg.inv(BODY_TO_RV)
    homo = np.concatenate([points_rv, np.ones((len(points_rv), 1), dtype=np.float64)], axis=1)
    pts_imu = (homo @ rv2imu.T)[:, :3]
    homo_i = np.concatenate([pts_imu, np.ones((len(pts_imu), 1), dtype=np.float64)], axis=1)
    return (homo_i @ imu2world.T)[:, :3]


def ego_yaw_from_pose(imu2world: np.ndarray) -> float:
    """Yaw of body x-forward in world XY (radians)."""
    return float(np.arctan2(imu2world[1, 0], imu2world[0, 0]))


def save_bev(
    path: Path,
    xyz: np.ndarray,
    intensity: np.ndarray,
    traj_xy: np.ndarray,
    traj_yaw: np.ndarray,
    resolution: float = 0.1,
    pad: float = 5.0,
    max_draw_pts: int = 2_000_000,
):
    """Top-down BEV (world XY) with intensity-colored points and ego trajectory."""
    xy = xyz[:, :2]
    all_xy = np.concatenate([xy, traj_xy], axis=0) if len(traj_xy) else xy
    xmin, ymin = all_xy.min(axis=0) - pad
    xmax, ymax = all_xy.max(axis=0) + pad
    w = int(np.ceil((xmax - xmin) / resolution)) + 1
    h = int(np.ceil((ymax - ymin) / resolution)) + 1
    max_side = 4000
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        resolution = resolution / scale
        w = int(np.ceil((xmax - xmin) / resolution)) + 1
        h = int(np.ceil((ymax - ymin) / resolution)) + 1

    if len(xy) > max_draw_pts:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xy), max_draw_pts, replace=False)
        xy = xy[sel]
        intensity = intensity[sel]

    bev = np.zeros((h, w), dtype=np.float32)
    col = ((xy[:, 0] - xmin) / resolution).astype(np.int32)
    row = ((xy[:, 1] - ymin) / resolution).astype(np.int32)
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    col, row = col[valid], row[valid]
    if intensity.max() > 1.5:
        inten = np.clip(intensity[valid] / 255.0, 0.0, 1.0)
    else:
        inten = np.clip(intensity[valid], 0.0, 1.0)
    # larger world-Y toward top of image
    row_img = (h - 1) - row
    flat = row_img * w + col
    order = np.argsort(inten)
    flat_s, inten_s = flat[order], inten[order]
    bev_flat = bev.ravel()
    bev_flat[flat_s] = inten_s
    bev = bev_flat.reshape(h, w)

    fig_w = max(8.0, w / 200.0)
    fig_h = max(8.0, h / 200.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=150)
    extent = [xmin, xmax, ymin, ymax]
    ax.imshow(bev, cmap="turbo", origin="upper", extent=extent, aspect="equal", vmin=0, vmax=1)

    if len(traj_xy) >= 2:
        pts = traj_xy
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0, 1, len(pts) - 1)
        lc = LineCollection(segs, cmap="spring", norm=plt.Normalize(0, 1), linewidths=2.5, zorder=5)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.plot(pts[0, 0], pts[0, 1], "o", color="lime", markersize=8, zorder=6, label="start")
        ax.plot(pts[-1, 0], pts[-1, 1], "s", color="red", markersize=8, zorder=6, label="end")
        step = max(1, len(traj_xy) // 12)
        arrow_len = max((xmax - xmin), (ymax - ymin)) * 0.025
        for i in range(0, len(traj_xy), step):
            yaw = traj_yaw[i]
            ax.arrow(
                traj_xy[i, 0], traj_xy[i, 1],
                arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
                head_width=arrow_len * 0.35, head_length=arrow_len * 0.3,
                fc="white", ec="black", lw=0.6, zorder=7, length_includes_head=True,
            )
        ax.legend(loc="upper right", framealpha=0.85)

    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title(f"BEV stacked renders + ego trajectory ({len(traj_xy)} poses)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gacrt_nvs_PCI_20260825.yaml")
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--raydrop_thresh", type=float, default=0.5)
    parser.add_argument("--max_pts", type=int, default=12_000_000)
    parser.add_argument("--bev_resolution", type=float, default=0.1, help="BEV meters/pixel")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    conf = OmegaConf.merge(
        OmegaConf.load(_ROOT / args.base_config),
        OmegaConf.load(_ROOT / args.config),
    )
    model_path = Path(conf.model_path)
    render_dir = model_path / "ray_drop_datasets" / f"render_{args.split}"
    if not render_dir.is_dir():
        raise SystemExit(f"Missing render dir: {render_dir}")

    scale_path = model_path / "scale_factor.txt"
    scale_factor = float(scale_path.read_text().strip()) if scale_path.is_file() else float(conf.scale_factor)

    json_path = Path(conf.source_path) / conf.sequence_id / f"transforms_{conf.sequence_id}_all.json"
    frames = json.loads(json_path.read_text())["frames"]
    pose_by_idx = {int(fr.get("frame_idx", i)): np.asarray(fr["lidar2world"], dtype=np.float64)
                   for i, fr in enumerate(frames)}

    pt_files = sorted(render_dir.glob("*.pt"), key=lambda p: int(p.stem))
    if not pt_files:
        raise SystemExit(f"No .pt files in {render_dir}")
    pt_files = pt_files[: args.n_frames]

    vfov = list(conf.vfov)
    xs, ys, zs, ints = [], [], [], []
    traj_xy, traj_yaw = [], []
    for pt_path in pt_files:
        frame_idx = int(pt_path.stem)
        if frame_idx not in pose_by_idx:
            print(f"[skip] no pose for frame {frame_idx}")
            continue
        T_w = pose_by_idx[frame_idx]
        traj_xy.append(T_w[:2, 3].copy())
        traj_yaw.append(ego_yaw_from_pose(T_w))

        try:
            pack = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            pack = torch.load(pt_path, map_location="cpu")
        raydrop = pack[0].numpy()
        intensity = pack[1].numpy()
        depth = pack[2].numpy()
        if depth.ndim == 2:
            depth = depth[None]
            intensity = intensity[None]
            raydrop = raydrop[None]
        mask_keep = (depth > 0) & (raydrop <= args.raydrop_thresh)
        depth_m = np.where(mask_keep, depth, 0.0).astype(np.float32)
        depth_m = depth_m / scale_factor

        pts_rv, mask = pano_to_xyz(depth_m, vfov, (-180.0, 180.0))
        if len(pts_rv) == 0:
            print(f"[skip] empty cloud frame {frame_idx}")
            continue
        inten = intensity[0][mask].astype(np.float32)
        pts_w = rv_to_world(pts_rv, T_w)

        aabb = getattr(conf, "ego_body_aabb", None)
        if aabb is not None:
            # Safety net: drop any residual floaters still inside the ego AABB
            # after RV GT-body paste (decode may still leave a few).
            pts_h = np.concatenate([pts_w, np.ones((len(pts_w), 1), dtype=np.float64)], axis=1)
            pts_imu = (pts_h @ np.linalg.inv(T_w).T)[:, :3]
            from utils.ego_body_mask import points_in_aabb_np
            keep = ~points_in_aabb_np(pts_imu, aabb)
            pts_w = pts_w[keep]
            inten = inten[keep]

        # Prefer GT AABB paste already baked into render_*.pt; optional rigid
        # template append is disabled by default (ego_body_append_template=false).
        if getattr(conf, "ego_body_append_template", False):
            body_path = model_path / "ego_body_imu.npz"
            if body_path.is_file():
                from utils.ego_body_mask import load_ego_body_cloud, append_ego_body_world_points
                body_xyz, body_i = load_ego_body_cloud(str(body_path))
                pts_w, inten = append_ego_body_world_points(pts_w, inten, body_xyz, body_i, T_w)

        if pts_w is None or len(pts_w) == 0:
            print(f"[skip] empty after ego_body filter frame {frame_idx}")
            continue

        xs.append(pts_w[:, 0].astype(np.float32))
        ys.append(pts_w[:, 1].astype(np.float32))
        zs.append(pts_w[:, 2].astype(np.float32))
        ints.append(np.clip(inten, 0.0, 1.0) * 255.0)
        print(f"frame {frame_idx:03d}: n={len(pts_w)} r_med={np.median(np.linalg.norm(pts_rv, axis=1)):.2f}m")

    if not xs:
        raise SystemExit("No points stacked")

    xyz = np.stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)], axis=1)
    inten = np.concatenate(ints)
    traj_xy = np.asarray(traj_xy, dtype=np.float64)
    traj_yaw = np.asarray(traj_yaw, dtype=np.float64)

    if len(xyz) > args.max_pts:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), args.max_pts, replace=False)
        xyz, inten = xyz[sel], inten[sel]

    stem = f"stacked_{args.split}_renders_first{len(pt_files)}_world"
    out = Path(args.out) if args.out else (model_path / f"{stem}.pcd")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_pcd(out, xyz, inten)
    print(f"wrote {out} n={len(xyz)} from {len(pt_files)} frames scale_factor={scale_factor}")

    bev_path = out.with_suffix(".bev.png")
    save_bev(bev_path, xyz, inten, traj_xy, traj_yaw, resolution=args.bev_resolution)
    print(f"wrote {bev_path} traj_len={len(traj_xy)} "
          f"span_xy=({np.ptp(traj_xy[:, 0]):.1f}m, {np.ptp(traj_xy[:, 1]):.1f}m)")


if __name__ == "__main__":
    main()
