#!/usr/bin/env python3
"""Ego-body replace via the real data pipeline: 3D replace → RV → decode PCD.

Correct flow (not stuffing template into the output PCD):
  1. In IMU frame, for each补盲 source: drop AABB body hits, append mesh/dense
     template points (same as replacing ego returns in the raw cloud).
  2. Project that cloud to panoramic RV with the existing blind-lidar camera
     model (front/back half + stitch), same as training GT construction.
  3. Decode RV → IMU points (pano_to_lidar + inv RT) and write PCD.
  4. Compare ORIG vs REPLACED on RV images and on RV-decoded PCDs.

Example:
  python scripts/visualize_ego_body_replace.py \\
    --config configs/gacrt_nvs_PCI_20260825_qt128.yaml \\
    --ego_cloud eval_output/.../ego_body_imu_clean_dense.pcd \\
    --aabb -1.4 1.4 -2 3.5 -1.8 0 \\
    --frames 0 50 100
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
import torch
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
    points_in_aabb,
    _backward_rt,
    _rt_from_w2c,
)
from utils.camera_utils import project_points_to_rv
from utils.graphics_utils import pano_to_lidar


def read_pcd_xyz_intensity(path: Path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            text = line.decode("ascii", errors="ignore").strip()
            header.append(text)
            if text.startswith("DATA"):
                break
        meta = {}
        for ln in header:
            if ln and not ln.startswith("#") and " " in ln:
                k, v = ln.split(" ", 1)
                meta[k] = v
        n = int(meta["POINTS"])
        fields = meta["FIELDS"].split()
        sizes = list(map(int, meta["SIZE"].split()))
        types = meta["TYPE"].split()
        raw = f.read()
        data_mode = meta["DATA"].split()[0]
    if data_mode != "binary":
        raise ValueError(f"Only binary PCD supported: {path}")
    dtype = []
    for name, sz, tp in zip(fields, sizes, types):
        dtype.append((name, {"F4": "<f4", "I4": "<i4"}[tp + str(sz)]))
    arr = np.frombuffer(raw, dtype=np.dtype(dtype), count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    inten = (arr["intensity"].astype(np.float32)
             if "intensity" in arr.dtype.names else np.ones(n, np.float32))
    return xyz, inten


def load_ego_template(path: Path, max_points: int = 300000):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        xyz = data["xyz"].astype(np.float32)
        inten = data["intensity"].astype(np.float32)
        if inten.max() <= 1.5:
            inten = inten * 255.0
    elif path.suffix.lower() == ".ply":
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            n = min(max_points, max(100000, len(mesh.vertices) * 20))
            pcd = mesh.sample_points_uniformly(number_of_points=int(n))
            xyz = np.asarray(pcd.points, dtype=np.float32)
        else:
            pcd = o3d.io.read_point_cloud(str(path))
            xyz = np.asarray(pcd.points, dtype=np.float32)
        inten = np.full(len(xyz), 128.0, dtype=np.float32)
    else:
        xyz, inten = read_pcd_xyz_intensity(path)
    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), max_points, replace=False)
        xyz, inten = xyz[sel], inten[sel]
    return xyz, inten


def write_pcd(path: Path, xyz, intensity=None, source=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    if n == 0:
        print(f"[warn] empty PCD skip {path}")
        return
    if intensity is None:
        intensity = np.ones(n, np.float32)
    if source is None:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["intensity"] = intensity.astype(np.float32)
        fields, size, types, count = "x y z intensity", "4 4 4 4", "F F F F", "1 1 1 1"
    else:
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("intensity", "<f4"), ("source", "<i4"),
        ])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["intensity"] = intensity.astype(np.float32)
        arr["source"] = source.astype(np.int32)
        fields, size, types, count = (
            "x y z intensity source", "4 4 4 4 4", "F F F F I", "1 1 1 1 1")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {fields}\nSIZE {size}\nTYPE {types}\nCOUNT {count}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def stitch_front_back(d_f, i_f, d_b, i_b):
    _, h, w = d_f.shape
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    depth = np.zeros((1, h, w * 2), np.float32)
    inten = np.zeros((1, h, w * 2), np.float32)
    depth[:, :, breaks[1]:breaks[2]] = d_f
    inten[:, :, breaks[1]:breaks[2]] = i_f
    depth[:, :, breaks[2]:breaks[3]] = d_b[:, :, 0:(breaks[3] - breaks[2])]
    depth[:, :, breaks[0]:breaks[1]] = d_b[:, :, (w - breaks[1] + breaks[0]):w]
    inten[:, :, breaks[2]:breaks[3]] = i_b[:, :, 0:(breaks[3] - breaks[2])]
    inten[:, :, breaks[0]:breaks[1]] = i_b[:, :, (w - breaks[1] + breaks[0]):w]
    return depth, inten


def points_to_half_rv(pts_imu, inten, rotation, translation, h, w, vfov, hfov):
    if len(pts_imu) == 0:
        z = np.zeros((1, h, w), np.float32)
        return z, z.copy()
    cam = (pts_imu.astype(np.float64) @ rotation.astype(np.float64)
           + translation.astype(np.float64)).astype(np.float32)
    d, i = project_points_to_rv(cam, inten, h, w, vfov, hfov)
    return d.astype(np.float32), i.astype(np.float32)


def project_cloud_to_rv(pts_imu, inten, R, T, h, w, vfov, hfov):
    """Existing training-style front/back panoramic projection."""
    R_b, T_b = _backward_rt(R, T)
    d_f, i_f = points_to_half_rv(pts_imu, inten, R, T, h, w, vfov, hfov)
    d_b, i_b = points_to_half_rv(pts_imu, inten, R_b, T_b, h, w, vfov, hfov)
    return stitch_front_back(d_f, i_f, d_b, i_b)


def rv_to_imu_points(depth, inten, R, T, vfov, hfov):
    """Decode stitched 360 RV → IMU (same model as project)."""
    _, h, w2 = depth.shape
    w = w2 // 2
    breaks = (0, w // 2, 3 * w // 2, w * 2)

    def half_from_stitch(pano):
        front = pano[:, :, breaks[1]:breaks[2]].copy()
        back = np.zeros_like(front)
        back[:, :, 0:(breaks[3] - breaks[2])] = pano[:, :, breaks[2]:breaks[3]]
        back[:, :, (w - breaks[1] + breaks[0]):w] = pano[:, :, breaks[0]:breaks[1]]
        return front, back

    d_f, d_b = half_from_stitch(depth)
    i_f, i_b = half_from_stitch(inten)
    R_b, T_b = _backward_rt(R, T)

    chunks_xyz, chunks_i = [], []
    for d_half, i_half, rot, trans in (
        (d_f, i_f, R, T),
        (d_b, i_b, R_b, T_b),
    ):
        pts_cam = pano_to_lidar(torch.from_numpy(d_half), vfov, hfov).cpu().numpy()
        if len(pts_cam) == 0:
            continue
        mask = d_half[0] > 0
        inten_flat = i_half[0][mask]
        inv_R = np.linalg.inv(rot.astype(np.float64))
        imu = (pts_cam.astype(np.float64) - trans.astype(np.float64)) @ inv_R
        chunks_xyz.append(imu.astype(np.float32))
        chunks_i.append(inten_flat.astype(np.float32))
    if not chunks_xyz:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(chunks_xyz, 0), np.concatenate(chunks_i, 0)


def replace_aabb_with_template(pts, inten, aabb, ego_xyz, ego_i):
    """Step 1: drop AABB hits, append template (IMU)."""
    body = points_in_aabb(pts, aabb)
    pts_out = np.concatenate([pts[~body], ego_xyz], axis=0)
    inten_out = np.concatenate([inten[~body], ego_i], axis=0)
    return pts_out, inten_out, int(body.sum())


def blind_rt(extrinsics, source_id):
    lidar_from_imu = np.linalg.inv(extrinsics[source_id]["imu_from_lidar"])
    w2c = BODY_TO_RV @ HESAI_TO_BODY @ lidar_from_imu
    return _rt_from_w2c(w2c)


def depth_vis(depth, vmax=None):
    d = depth[0].copy()
    if vmax is None:
        pos = d[d > 0]
        vmax = float(np.percentile(pos, 98)) if len(pos) else 1.0
    vmax = max(vmax, 1e-3)
    img = np.clip(d / vmax, 0, 1)
    img[d <= 0] = 0
    return img, vmax


def main():
    parser = argparse.ArgumentParser(
        description="3D AABB replace → RV project → RV decode PCD")
    parser.add_argument("--config", type=str,
                        default="configs/gacrt_nvs_PCI_20260825_qt128.yaml")
    parser.add_argument(
        "--ego_cloud", type=str,
        default="eval_output/gacrt_reconstruction/PCI_20260825_qt128/"
                "ego_body_imu_clean_dense.pcd")
    parser.add_argument(
        "--aabb", type=float, nargs=6, default=[-1.4, 1.4, -2.0, 3.5, -1.8, 0.0],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100])
    parser.add_argument("--sources", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--out_dir", type=str,
        default="eval_output/gacrt_reconstruction/PCI_20260825_qt128/"
                "ego_replace_vis")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    data_root = Path(conf.source_path)
    transforms = json.loads(
        (data_root / conf.sequence_id /
         f"transforms_{conf.sequence_id}_all.json").read_text())
    frames_meta = {int(f["frame_idx"]): f for f in transforms["frames"]}
    extrinsics = load_blind_extrinsics(conf.calib_dir)
    aabb = np.asarray(args.aabb, dtype=np.float64).reshape(3, 2)
    h, w = int(conf.hw[0]), int(conf.hw[1])
    vfov, hfov = list(conf.vfov), list(conf.hfov)

    ego_xyz, ego_i = load_ego_template(args.ego_cloud)
    keep = points_in_aabb(ego_xyz, aabb)
    ego_xyz, ego_i = ego_xyz[keep], ego_i[keep]
    print(f"[vis] pipeline: replace AABB → project RV → decode PCD")
    print(f"[vis] ego template n={len(ego_xyz)} aabb=\n{aabb}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name_by_id = {sid: name for sid, name in BLIND_LIDARS}

    for frame_idx in args.frames:
        if frame_idx not in frames_meta:
            print(f"[skip] frame_idx={frame_idx}")
            continue
        fr = frames_meta[frame_idx]
        path = data_root / fr["lidar_path"]
        xyz, inten, source = load_lidar_xyz_intensity(
            str(path), n_feats=int(fr.get("num_pts_feats", 6)), return_source=True)
        finite = np.isfinite(xyz).all(1)
        xyz, inten, source = xyz[finite], inten[finite], source[finite]

        fig_rows = []
        merge_o_xyz, merge_o_i, merge_o_s = [], [], []
        merge_r_xyz, merge_r_i, merge_r_s = [], [], []

        for source_id in args.sources:
            if source_id not in extrinsics:
                continue
            sel = source == source_id
            if not np.any(sel):
                continue
            pts = xyz[sel]
            its = inten[sel]
            R, T = blind_rt(extrinsics, source_id)

            # --- step 1: 3D replace in AABB ---
            pts_rep, its_rep, n_body = replace_aabb_with_template(
                pts, its, aabb, ego_xyz, ego_i)

            # --- step 2: project to RV (existing flow) ---
            d_o, i_o = project_cloud_to_rv(pts, its, R, T, h, w, vfov, hfov)
            d_r, i_r = project_cloud_to_rv(pts_rep, its_rep, R, T, h, w, vfov, hfov)

            # --- step 3: RV → IMU PCD ---
            pts_o, its_o = rv_to_imu_points(d_o, i_o, R, T, vfov, hfov)
            pts_r, its_r = rv_to_imu_points(d_r, i_r, R, T, vfov, hfov)

            name = name_by_id.get(source_id, str(source_id))
            write_pcd(out_dir / f"frame{frame_idx:03d}_{name}_from_rv_orig.pcd",
                      pts_o, its_o)
            write_pcd(out_dir / f"frame{frame_idx:03d}_{name}_from_rv_replaced.pcd",
                      pts_r, its_r)
            np.savez_compressed(
                out_dir / f"frame{frame_idx:03d}_{name}_rv.npz",
                depth_orig=d_o, inten_orig=i_o,
                depth_replaced=d_r, inten_replaced=i_r)

            merge_o_xyz.append(pts_o); merge_o_i.append(its_o)
            merge_o_s.append(np.full(len(pts_o), source_id, np.int32))
            merge_r_xyz.append(pts_r); merge_r_i.append(its_r)
            merge_r_s.append(np.full(len(pts_r), source_id, np.int32))

            vmax = max(
                np.percentile(d_o[d_o > 0], 98) if np.any(d_o > 0) else 1.0,
                np.percentile(d_r[d_r > 0], 98) if np.any(d_r > 0) else 1.0,
            )
            vo, _ = depth_vis(d_o, vmax)
            vr, _ = depth_vis(d_r, vmax)
            vd = np.abs(vo - vr)
            io = np.clip(i_o[0] / max(float(i_o[0].max()), 1e-3), 0, 1)
            ir = np.clip(i_r[0] / max(float(i_r[0].max()), 1e-3), 0, 1)
            fig_rows.append((
                name, vo, vr, vd, io, ir, n_body,
                int((d_o > 0).sum()), int((d_r > 0).sum()),
                len(pts_o), len(pts_r)))

        if not fig_rows:
            print(f"[warn] frame {frame_idx}: no blind sources")
            continue

        # Merged RV-decoded clouds (the outputs to visualize / diff)
        write_pcd(
            out_dir / f"frame{frame_idx:03d}_from_rv_orig.pcd",
            np.concatenate(merge_o_xyz), np.concatenate(merge_o_i),
            np.concatenate(merge_o_s))
        write_pcd(
            out_dir / f"frame{frame_idx:03d}_from_rv_replaced.pcd",
            np.concatenate(merge_r_xyz), np.concatenate(merge_r_i),
            np.concatenate(merge_r_s))

        n = len(fig_rows)
        fig, axes = plt.subplots(n, 5, figsize=(18, 2.2 * n), squeeze=False)
        for r, row in enumerate(fig_rows):
            name, vo, vr, vd, io, ir, n_body, px_o, px_r, n_o, n_r = row
            axes[r, 0].imshow(vo, cmap="turbo", vmin=0, vmax=1)
            axes[r, 0].set_title(f"{name} depth ORIG body3d={n_body} px={px_o}")
            axes[r, 1].imshow(vr, cmap="turbo", vmin=0, vmax=1)
            axes[r, 1].set_title(f"{name} depth REPLACED→RV px={px_r}")
            axes[r, 2].imshow(vd, cmap="magma", vmin=0, vmax=1)
            axes[r, 2].set_title("|Δ depth|")
            axes[r, 3].imshow(io, cmap="gray", vmin=0, vmax=1)
            axes[r, 3].set_title(f"inten ORIG | decode n={n_o}")
            axes[r, 4].imshow(ir, cmap="gray", vmin=0, vmax=1)
            axes[r, 4].set_title(f"inten REPLACED | decode n={n_r}")
            for c in range(5):
                axes[r, c].axis("off")
        fig.suptitle(
            f"frame={frame_idx}: AABB replace → RV project → RV decode",
            fontsize=12)
        fig.tight_layout()
        png = out_dir / f"frame{frame_idx:03d}_rv_compare.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"[vis] frame {frame_idx}: {png.name} + from_rv_*.pcd")

    print(f"[vis] done → {out_dir.resolve()}")
    print("[vis] compare: frameXXX_from_rv_orig.pcd vs frameXXX_from_rv_replaced.pcd")


if __name__ == "__main__":
    main()
