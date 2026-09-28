#!/usr/bin/env python3
"""Colorize blind-lidar clouds by KEEP_YAW only and dump PCD + RV.

Labels (RGB):
  keep  green  — yaw in suggested KEEP_YAW interval
  drop  red    — yaw outside KEEP_YAW

Example:
  python scripts/visualize_blind_azimuth_keep.py --n_frames 10
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
    BODY_TO_RV,
    HESAI_TO_BODY,
    load_blind_extrinsics,
    load_lidar_xyz_intensity,
    _backward_rt,
    _rt_from_w2c,
)

# Training FoV keep intervals in lidar-body yaw (deg): all blinds use ±90°
# (one 180° half each; no front/back 360 stitch).
KEEP_YAW = {
    "left": {"intervals": [(-90.0, 90.0)], "wraps": False},
    "right": {"intervals": [(-90.0, 90.0)], "wraps": False},
    "front": {"intervals": [(-90.0, 90.0)], "wraps": False},
    "back": {"intervals": [(-90.0, 90.0)], "wraps": False},
}

COLORS = {
    "keep": (0, 200, 80),   # green — inside KEEP_YAW
    "drop": (230, 60, 40),  # red   — outside KEEP_YAW
}

LABEL_ID = {k: i for i, k in enumerate(COLORS)}


def in_intervals(az_deg, intervals):
    ok = np.zeros(len(az_deg), dtype=bool)
    for lo, hi in intervals:
        ok |= (az_deg >= lo) & (az_deg < hi)
    return ok


def to_lidar_body(pts_imu, imu_from_lidar):
    lidar_from_imu = np.linalg.inv(imu_from_lidar)
    homo = np.c_[pts_imu.astype(np.float64), np.ones(len(pts_imu))]
    p_lidar = (homo @ lidar_from_imu.T)[:, :3]
    homo2 = np.c_[p_lidar, np.ones(len(p_lidar))]
    return (homo2 @ HESAI_TO_BODY.T)[:, :3].astype(np.float32)


def write_pcd_rgb(path: Path, xyz, rgb, label=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    if label is None:
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        fields = "x y z red green blue"
        size, types, count = "4 4 4 1 1 1", "F F F U U U", "1 1 1 1 1 1"
    else:
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("label", "<i4"),
        ])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        arr["label"] = label.astype(np.int32)
        fields = "x y z red green blue label"
        size, types, count = "4 4 4 1 1 1 4", "F F F U U U I", "1 1 1 1 1 1 1"
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {fields}\nSIZE {size}\nTYPE {types}\nCOUNT {count}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def classify_points(pts_imu, imu_from_lidar, keep_cfg):
    """Color only by KEEP_YAW: keep=green, drop=red."""
    p_body = to_lidar_body(pts_imu, imu_from_lidar)
    az = np.arctan2(p_body[:, 1], p_body[:, 0]) * 180.0 / np.pi
    keep = in_intervals(az, keep_cfg["intervals"])
    labels = np.where(keep, "keep", "drop")
    rgb = np.zeros((len(pts_imu), 3), dtype=np.uint8)
    lid = np.zeros(len(pts_imu), dtype=np.int32)
    for name, color in COLORS.items():
        m = labels == name
        rgb[m] = color
        lid[m] = LABEL_ID[name]
    return labels, rgb, lid, az, p_body


def stitch_front_back(d_f, i_f, d_b, i_b):
    _, h, w = d_f.shape
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    depth = np.zeros((1, h, w * 2), np.float32)
    inten = np.zeros((1, h, w * 2), np.float32)
    depth[:, :, breaks[1]:breaks[2]] = d_f
    inten[:, :, breaks[1]:breaks[2]] = i_f
    depth[:, :, breaks[2]:breaks[3]] = d_b[:, :, : breaks[3] - breaks[2]]
    depth[:, :, breaks[0]:breaks[1]] = d_b[:, :, (w - breaks[1] + breaks[0]): w]
    inten[:, :, breaks[2]:breaks[3]] = i_b[:, :, : breaks[3] - breaks[2]]
    inten[:, :, breaks[0]:breaks[1]] = i_b[:, :, (w - breaks[1] + breaks[0]): w]
    return depth, inten


def project_label_rv(pts_imu, labels, R, T, h, w, vfov, hfov):
    """Project with nearest depth; paint RGB from winning label."""
    R_b, T_b = _backward_rt(R, T)
    # encode label id as fake intensity then map back — better: custom z-buffer
    depth = np.zeros((1, h, w * 2), np.float32)
    rgb = np.zeros((h, w * 2, 3), dtype=np.uint8)
    lab = np.full((h, w * 2), -1, dtype=np.int32)

    def paint_half(pts, labs, rot, trans, col_slice):
        if len(pts) == 0:
            return
        cam = pts.astype(np.float64) @ rot.astype(np.float64) + trans.astype(np.float64)
        # reuse projector logic for u,v,r
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        phi = np.arctan2(x, z)
        theta = np.arctan2(np.sqrt(x * x + z * z), -y)
        r = np.sqrt(x * x + y * y + z * z)
        vfov_max = np.pi / 2 - vfov[0] * np.pi / 180
        vfov_min = np.pi / 2 - vfov[1] * np.pi / 180
        hfov_max = hfov[1] * np.pi / 180
        hfov_min = hfov[0] * np.pi / 180
        row = (theta - vfov_min) * h / (vfov_max - vfov_min)
        col = (phi - hfov_min) * w / (hfov_max - hfov_min)
        m = (row >= -0.5) & (row < h - 0.5) & (col >= -0.5) & (col < w - 0.5) & (r > 0)
        row = np.around(row[m]).astype(int)
        col = np.around(col[m]).astype(int)
        rr, labs_m = r[m], labs[m]
        # map half columns into stitched panorama
        # front: breaks[1]:breaks[2]; back: remap later via caller slices
        for i in range(len(row)):
            c = col_slice(col[i])
            if c is None:
                continue
            ri, ci = row[i], c
            if depth[0, ri, ci] == 0 or rr[i] < depth[0, ri, ci]:
                depth[0, ri, ci] = rr[i]
                name = labs_m[i]
                rgb[ri, ci] = COLORS[name]
                lab[ri, ci] = LABEL_ID[name]

    _, _, w_half = 1, h, w
    breaks = (0, w // 2, 3 * w // 2, w * 2)

    def front_col(c):
        return breaks[1] + c

    def back_col(c):
        # inverse of stitch: back left part -> right of pano, back right -> left
        if c < (breaks[3] - breaks[2]):
            return breaks[2] + c
        # c in second part of back half maps to left strip
        # back[:, (w - breaks[1]):w] -> breaks[0]:breaks[1]
        # index in back half: offset = c - (w - breaks[1]) ... use same as stitch inverse
        left_w = breaks[1] - breaks[0]
        # columns that go to left strip are the last left_w of back
        if c >= w - left_w:
            return breaks[0] + (c - (w - left_w))
        return None

    paint_half(pts_imu, labels, R, T, front_col)
    paint_half(pts_imu, labels, R_b, T_b, back_col)
    return depth, rgb, lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gacrt_nvs_PCI_20260825_qt128.yaml")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        default="eval_output/gacrt_reconstruction/PCI_20260825_qt128/"
                "ego_azimuth_stats/frame_vis_keep_yaw")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    data_root = Path(conf.source_path)
    transforms = json.loads(
        (data_root / conf.sequence_id /
         f"transforms_{conf.sequence_id}_all.json").read_text())
    frames = transforms["frames"]
    idxs = np.linspace(args.start_frame, len(frames) - 1, args.n_frames, dtype=int)
    extrinsics = load_blind_extrinsics(conf.calib_dir)
    h, w = int(conf.hw[0]), int(conf.hw[1])
    vfov, hfov = list(conf.vfov), list(conf.hfov)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    legend = " | ".join(f"{k}={v}" for k, v in COLORS.items())
    (out_dir / "legend.txt").write_text(
        legend + "\n" + json.dumps(KEEP_YAW, indent=2) + "\n")

    for fi in idxs:
        fr = frames[int(fi)]
        frame_idx = int(fr.get("frame_idx", fi))
        path = data_root / fr["lidar_path"]
        xyz, inten, source = load_lidar_xyz_intensity(
            str(path), n_feats=int(fr.get("num_pts_feats", 6)), return_source=True)
        finite = np.isfinite(xyz).all(1)
        xyz, inten, source = xyz[finite], inten[finite], source[finite]

        merge_xyz, merge_rgb, merge_lab = [], [], []
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))

        for row, (sid, name) in enumerate(BLIND_LIDARS):
            sel = source == sid
            ax0, ax1 = axes[row, 0], axes[row, 1]
            if not np.any(sel):
                ax0.set_title(f"{name}: no points")
                ax1.axis("off")
                ax0.axis("off")
                continue
            pts = xyz[sel]
            keep_cfg = KEEP_YAW[name]
            labels, rgb, lid, az, p_body = classify_points(
                pts, extrinsics[sid]["imu_from_lidar"], keep_cfg)

            write_pcd_rgb(
                out_dir / f"frame{frame_idx:03d}_{name}_imu_colored.pcd",
                pts, rgb, lid)
            write_pcd_rgb(
                out_dir / f"frame{frame_idx:03d}_{name}_lidar_body_colored.pcd",
                p_body, rgb, lid)

            merge_xyz.append(pts)
            merge_rgb.append(rgb)
            merge_lab.append(lid)

            lidar_from_imu = np.linalg.inv(extrinsics[sid]["imu_from_lidar"])
            w2c = BODY_TO_RV @ HESAI_TO_BODY @ lidar_from_imu
            R, T = _rt_from_w2c(w2c)
            depth, rv_rgb, _ = project_label_rv(
                pts, labels, R, T, h, w, vfov, hfov)
            d = depth[0]
            vmax = np.percentile(d[d > 0], 98) if np.any(d > 0) else 1.0
            d_img = np.clip(d / max(vmax, 1e-3), 0, 1)
            ax0.imshow(d_img, cmap="turbo", vmin=0, vmax=1, aspect="auto")
            ax0.set_title(f"{name} depth RV")
            ax0.axis("off")
            ax1.imshow(rv_rgb, aspect="auto")
            ax1.set_title(f"{name} KEEP_YAW {keep_cfg['intervals']}")
            ax1.axis("off")
            plt.imsave(
                out_dir / f"frame{frame_idx:03d}_{name}_rv_label.png", rv_rgb)

            counts = {k: int((labels == k).sum()) for k in COLORS}
            print(f"frame {frame_idx} {name}: {counts}")

        if merge_xyz:
            write_pcd_rgb(
                out_dir / f"frame{frame_idx:03d}_all_blind_imu_colored.pcd",
                np.concatenate(merge_xyz), np.concatenate(merge_rgb),
                np.concatenate(merge_lab))

        fig.suptitle(
            f"frame {frame_idx} | green=KEEP_YAW keep | red=drop",
            fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"frame{frame_idx:03d}_rv_grid.png", dpi=140)
        plt.close(fig)
        print(f"[vis] wrote frame {frame_idx}")

    print(f"[vis] done → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
