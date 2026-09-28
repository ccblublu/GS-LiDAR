#!/usr/bin/env python3
"""Preview final QT128 blind training input for a few frames.

Mirrors loader + loadCam projection (cam_num=1, hfov ±90):
  - scene points (ego AABB removed) → pts_depth / pts_intensity
  - ego_body_mask (ignore holes)

Example:
  python scripts/preview_blind_train_input.py --n_frames 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gacrt_loader import (
    BLIND_LIDARS,
    build_blind_frame_cameras,
    load_blind_extrinsics,
    load_lidar_xyz_intensity,
)
from utils.camera_utils import project_points_to_rv

SLOT_NAME = {i: name for i, (_, name) in enumerate(BLIND_LIDARS)}
SOURCE_NAME = {sid: name for sid, name in BLIND_LIDARS}


def depth_rgb(depth_1hw, near=0.5, far=60.0):
    d = np.asarray(depth_1hw[0], dtype=np.float32).copy()
    valid = d > 0
    out = np.zeros((*d.shape, 3), dtype=np.float32)
    if not valid.any():
        return out
    # log-ish turbo-like via matplotlib
    x = d[valid]
    x = np.clip(x, near, far)
    x = (np.log(x) - np.log(near)) / (np.log(far) - np.log(near) + 1e-8)
    cmap = plt.get_cmap("turbo")
    out[valid] = cmap(x)[:, :3]
    return out


def inten_rgb(inten_1hw):
    i = np.asarray(inten_1hw[0], dtype=np.float32)
    i = np.clip(i, 0, 1)
    return np.stack([i, i, i], axis=-1)


def mask_rgb(mask_1hw):
    m = np.asarray(mask_1hw)
    if m.ndim == 3:
        m = m[0]
    m = (m > 0.5).astype(np.float32)
    rgb = np.zeros((*m.shape, 3), dtype=np.float32)
    rgb[..., 0] = m  # red = body ignore
    return rgb


def overlay_depth_mask(depth_rgb_img, mask_1hw, alpha=0.45):
    m = np.asarray(mask_1hw)
    if m.ndim == 3:
        m = m[0]
    body = m > 0.5
    out = depth_rgb_img.copy()
    out[body] = (1 - alpha) * out[body] + alpha * np.array([1.0, 0.15, 0.1], dtype=np.float32)
    return out


def rv_to_xyz(depth_1hw, vfov, hfov):
    """Decode HxW range image with given FoV back to camera-frame xyz."""
    d = np.asarray(depth_1hw[0], dtype=np.float32)
    h, w = d.shape
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    vf0, vf1 = float(vfov[0]), float(vfov[1])
    hf0, hf1 = float(hfov[0]), float(hfov[1])
    VFOV_max = np.pi / 2 - vf0 * np.pi / 180
    VFOV_min = np.pi / 2 - vf1 * np.pi / 180
    HFOV_max = hf1 * np.pi / 180
    HFOV_min = hf0 * np.pi / 180
    theta = rows / h * (VFOV_max - VFOV_min) + VFOV_min
    phi = cols / w * (HFOV_max - HFOV_min) + HFOV_min
    # inverse of project_points_to_rv angles
    # theta = arctan2(sqrt(x^2+z^2), -y); phi = arctan2(x, z)
    # y = -r cos(theta); horiz = r sin(theta); x = horiz sin(phi); z = horiz cos(phi)
    r = d
    valid = r > 0
    horiz = r * np.sin(theta)
    x = horiz * np.sin(phi)
    z = horiz * np.cos(phi)
    y = -r * np.cos(theta)
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)
    return pts


def write_pcd(path: Path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.full((n, 3), 200, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb, dtype=np.uint8)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
        for i in range(n):
            packed = (int(rgb[i, 0]) << 16) | (int(rgb[i, 1]) << 8) | int(rgb[i, 2])
            f.write(f"{xyz[i, 0]:.4f} {xyz[i, 1]:.4f} {xyz[i, 2]:.4f} {packed}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gacrt_nvs_PCI_20260825_qt128.yaml")
    ap.add_argument("--base_config", default="configs/base.yaml")
    ap.add_argument("--n_frames", type=int, default=3)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args_cli = ap.parse_args()

    conf = OmegaConf.merge(
        OmegaConf.load(ROOT / args_cli.base_config),
        OmegaConf.load(ROOT / args_cli.config),
    )
    seq = conf.sequence_id
    source_path = Path(conf.source_path)
    data_root = source_path / seq
    json_path = data_root / f"transforms_{seq}_all.json"
    with open(json_path) as f:
        meta = json.load(f)
    poses = meta["frames"]
    n_take = min(args_cli.n_frames, len(poses) - args_cli.start_frame)

    out_dir = Path(args_cli.out_dir) if args_cli.out_dir else (
        Path(conf.model_path) / "train_input_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    extrinsics = load_blind_extrinsics(conf.calib_dir)
    h, w = int(conf.hw[0]), int(conf.hw[1])
    vfov, hfov = list(conf.vfov), list(conf.hfov)
    frames_total = len(poses)

    summary = {
        "config": args_cli.config,
        "cam_num": int(conf.cam_num),
        "vfov": vfov,
        "hfov": hfov,
        "hw": [h, w],
        "frames": [],
    }

    print(f"Preview {n_take} frames → {out_dir}")
    print(f"  cam_num={conf.cam_num} hfov={hfov} vfov={vfov} hw=[{h},{w}]")

    for local_i in range(n_take):
        frame_idx = args_cli.start_frame + local_i
        pose_info = poses[frame_idx]
        lidar_path = pose_info["lidar_path"]
        full_path = source_path / lidar_path
        n_feats = pose_info.get("num_pts_feats", 5)
        lidar2global = np.array(pose_info["lidar2world"], dtype=np.float64)

        points, intensity, source = load_lidar_xyz_intensity(
            str(full_path), n_feats, return_source=True)
        intensity = intensity / 255.0

        cams, init_world, n_body, n_sup, body_imu, _ = build_blind_frame_cameras(
            points, intensity, source, lidar2global, extrinsics, conf,
            frame_idx, frames_total, timestamp=float(frame_idx), lidar_path=lidar_path,
        )

        frame_dir = out_dir / f"frame_{frame_idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # raw source counts
        src_counts = {SOURCE_NAME.get(int(s), str(int(s))): int((source == s).sum())
                      for s in np.unique(source)}
        frame_rec = {
            "frame_idx": frame_idx,
            "lidar_path": lidar_path,
            "n_raw": int(len(points)),
            "source_counts": src_counts,
            "n_body_aabb": int(n_body),
            "n_supervise_world": int(n_sup),
            "n_init_world": int(len(init_world)),
            "n_cameras": len(cams),
            "cameras": [],
        }

        n_sensors = max(1, len(cams))
        fig, axes = plt.subplots(n_sensors, 4, figsize=(16, 2.4 * n_sensors), squeeze=False)
        for row, cam in enumerate(cams):
            name = SLOT_NAME.get(int(cam.sensor_id), f"slot{cam.sensor_id}")
            depth, inten = project_points_to_rv(
                cam.pointcloud_camera, cam.intensity, h, w, vfov, hfov)
            ego = cam.ego_body_mask
            if ego is None:
                ego = np.zeros((1, h, w), dtype=np.float32)
            elif np.asarray(ego).ndim == 2:
                ego = np.asarray(ego)[None]

            cov = float((depth[0] > 0).mean())
            body_cov = float((ego[0] > 0.5).mean())
            n_pts = int(len(cam.pointcloud_camera)) if cam.pointcloud_camera is not None else 0
            n_body_cam = (0 if cam.ego_body_points_cam is None
                         else int(len(cam.ego_body_points_cam)))

            d_rgb = depth_rgb(depth)
            i_rgb = inten_rgb(inten)
            m_rgb = mask_rgb(ego)
            o_rgb = overlay_depth_mask(d_rgb, ego)

            for ax, img, title in zip(
                axes[row],
                [d_rgb, i_rgb, m_rgb, o_rgb],
                ["depth (scene GT)", "intensity", "ego_body_mask", "depth+mask"],
            ):
                ax.imshow(img, aspect="auto")
                ax.set_title(f"{name}/{cam.towards} | {title}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

            # save per-sensor npy + decoded pcd
            np.savez_compressed(
                frame_dir / f"{name}_rv.npz",
                depth=depth.astype(np.float32),
                intensity=inten.astype(np.float32),
                ego_body_mask=np.asarray(ego, dtype=np.float32),
            )
            xyz = rv_to_xyz(depth, vfov, hfov)
            write_pcd(frame_dir / f"{name}_decoded_scene.pcd", xyz)

            # body points in cam (if any)
            if cam.ego_body_points_cam is not None and len(cam.ego_body_points_cam):
                write_pcd(
                    frame_dir / f"{name}_body_cam.pcd",
                    cam.ego_body_points_cam,
                    rgb=np.full((len(cam.ego_body_points_cam), 3), [230, 40, 40], dtype=np.uint8),
                )

            frame_rec["cameras"].append({
                "name": name,
                "sensor_id": int(cam.sensor_id),
                "towards": cam.towards,
                "n_scene_pts_cam": n_pts,
                "n_body_pts_cam": n_body_cam,
                "depth_coverage": cov,
                "ego_mask_coverage": body_cov,
                "depth_shape": list(depth.shape),
                "uid": int(cam.uid),
            })
            print(
                f"  frame {frame_idx} {name}: scene_pts={n_pts} body_pts={n_body_cam} "
                f"depth_cov={cov:.1%} mask_cov={body_cov:.1%} shape={depth.shape}"
            )

        fig.suptitle(
            f"frame {frame_idx} | cam_num={conf.cam_num} hfov={hfov} | "
            f"final training GT RV (ego body removed from depth)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(frame_dir / "overview.png", dpi=140)
        plt.close(fig)

        # also dump init world cloud (what seeds Gaussians for this frame chunk)
        if len(init_world):
            write_pcd(frame_dir / "init_world_scene.pcd", init_world)

        summary["frames"].append(frame_rec)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Open e.g. {out_dir / 'frame_0000' / 'overview.png'}")


if __name__ == "__main__":
    main()
