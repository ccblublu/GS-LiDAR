#!/usr/bin/env python3
"""Accumulate ego-body LiDAR hits across frames into one IMU-frame PCD.

Point clouds in PCI / GACRT transforms are already in the IMU / ego frame
(x-forward, y-left, z-up). Body returns are therefore rigidly aligned across
frames and can be stacked without transforming into a lidar sensor frame.

Example:
  python scripts/build_ego_body_cloud.py \\
    --transforms data/PCI_20260825/transforms_PCI_20260825_all.json \\
    --n_frames 200 \\
    --out eval_output/gacrt_reconstruction/PCI_20260825_qt128/ego_body_imu_accum200.pcd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gacrt_loader import load_lidar_xyz_intensity, points_in_aabb


# Config baseline (qt128) expanded for visualization tuning.
# IMU z≈0 is near sensor height; ground returns sit around z≈-1.9 m.
DEFAULT_AABB = [
    [-1.4, 1.4],   # was [-1.20, 1.20]
    [-2, 3.5],   # was [-1.70, 4.55]
    [-1.8, 0],   # was [-0.20, 2.20]; include ground + lower body
]
DEFAULT_SOURCES = [1, 2, 3, 4]  # QT128 补盲; exclude top unless --sources all


def parse_aabb(values):
    if values is None:
        return np.asarray(DEFAULT_AABB, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(3, 2)
    return arr


def resolve_lidar_path(lidar_path: str, data_root: Path) -> Path:
    p = Path(lidar_path)
    if p.is_file():
        return p
    cand = data_root / lidar_path
    if cand.is_file():
        return cand
    cand = data_root / "raw_data" / Path(lidar_path).name
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"Cannot resolve lidar path: {lidar_path} (data_root={data_root})")


def voxel_downsample(xyz, intensity, source, voxel):
    if voxel is None or voxel <= 0 or len(xyz) == 0:
        return xyz, intensity, source
    keys = np.floor(xyz / float(voxel)).astype(np.int64)
    # unique rows
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    src_out = source[idx] if source is not None else None
    return xyz[idx], intensity[idx], src_out


def write_pcd_binary(path: Path, xyz, intensity, source=None):
    """Write binary PCD in IMU / ego frame (fields: x y z intensity [source])."""
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    if source is None:
        fields = "x y z intensity"
        size = "4 4 4 4"
        types = "F F F F"
        count = "1 1 1 1"
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
        ])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["intensity"] = intensity.astype(np.float32)
    else:
        fields = "x y z intensity source"
        size = "4 4 4 4 4"
        types = "F F F F I"
        count = "1 1 1 1 1"
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("intensity", "<f4"), ("source", "<i4"),
        ])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["intensity"] = intensity.astype(np.float32)
        arr["source"] = source.astype(np.int32)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {size}\n"
        f"TYPE {types}\n"
        f"COUNT {count}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Stack ego-body hits into IMU-frame PCD")
    parser.add_argument(
        "--transforms", type=str,
        default="data/PCI_20260825/transforms_PCI_20260825_all.json")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--n_frames", type=int, default=1000,
                        help="Use the first N frames in transforms order")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Skip frames with frame_idx < start_frame")
    parser.add_argument(
        "--aabb", type=float, nargs=6, default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Relaxed IMU-frame AABB (default: slightly larger than qt128 config)")
    parser.add_argument(
        "--sources", type=str, default="1,2,3,4",
        help="Comma-separated source ids, or 'all'")
    parser.add_argument("--voxel", type=float, default=0,
                        help="Voxel size (m) after stacking; 0 disables")
    parser.add_argument("--max_points", type=int, default=0,
                        help="Random subsample cap after voxel (0 = no cap)")
    parser.add_argument(
        "--out", type=str,
        default="eval_output/gacrt_reconstruction/PCI_20260825_qt128/"
                "ego_body_imu_accum200.pcd")
    parser.add_argument("--also_npz", action="store_true",
                        help="Also write ego_body_imu.npz next to the PCD")
    args = parser.parse_args()

    transforms_path = Path(args.transforms)
    data_root = Path(args.data_root)
    meta = json.loads(transforms_path.read_text())
    frames = list(meta["frames"])
    frames = [f for f in frames if int(f.get("frame_idx", 0)) >= args.start_frame]
    frames = frames[: args.n_frames]
    if not frames:
        raise SystemExit("No frames selected")

    aabb = parse_aabb(args.aabb)
    if args.sources.strip().lower() == "all":
        source_ids = None
    else:
        source_ids = [int(x) for x in args.sources.split(",") if x.strip() != ""]

    print(f"[ego accum] frames={len(frames)} "
          f"(frame_idx {frames[0].get('frame_idx')}..{frames[-1].get('frame_idx')})")
    print(f"[ego accum] aabb (IMU)=\n{aabb}")
    print(f"[ego accum] sources={source_ids if source_ids is not None else 'all'}")

    chunks_xyz, chunks_i, chunks_s = [], [], []
    per_frame = []
    for fr in frames:
        path = resolve_lidar_path(fr["lidar_path"], data_root)
        n_feats = int(fr.get("num_pts_feats", 6))
        xyz, inten, source = load_lidar_xyz_intensity(
            str(path), n_feats=n_feats, return_source=True)
        finite = np.isfinite(xyz).all(axis=1)
        xyz, inten = xyz[finite], inten[finite]
        if source is not None:
            source = source[finite]

        if source_ids is not None:
            if source is None:
                raise RuntimeError(f"No source channel in {path}")
            sel = np.isin(source, np.asarray(source_ids, dtype=np.int32))
            xyz, inten, source = xyz[sel], inten[sel], source[sel]

        if len(xyz) == 0:
            per_frame.append(0)
            continue
        inside = points_in_aabb(xyz, aabb)
        n = int(inside.sum())
        per_frame.append(n)
        if n == 0:
            continue
        chunks_xyz.append(xyz[inside])
        chunks_i.append(inten[inside])
        if source is not None:
            chunks_s.append(source[inside])

    if not chunks_xyz:
        raise SystemExit("No ego-body points found — widen --aabb or check --sources")

    xyz = np.concatenate(chunks_xyz, axis=0)
    inten = np.concatenate(chunks_i, axis=0)
    source = np.concatenate(chunks_s, axis=0) if chunks_s else None
    print(f"[ego accum] stacked raw n={len(xyz)} "
          f"(mean body/frame={np.mean(per_frame):.1f}, "
          f"frames_with_hits={sum(x > 0 for x in per_frame)})")

    xyz, inten, source = voxel_downsample(xyz, inten, source, args.voxel)
    print(f"[ego accum] after voxel={args.voxel} n={len(xyz)}")

    if args.max_points > 0 and len(xyz) > args.max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), args.max_points, replace=False)
        xyz, inten = xyz[sel], inten[sel]
        if source is not None:
            source = source[sel]
        print(f"[ego accum] subsampled to {len(xyz)}")

    # Intensity may be 0..255; keep raw for visualization.
    out = Path(args.out)
    write_pcd_binary(out, xyz, inten, source)
    print(f"[ego accum] wrote IMU-frame PCD: {out.resolve()} n={len(xyz)}")
    print(f"[ego accum] bbox xyz min={xyz.min(0)} max={xyz.max(0)}")

    if args.also_npz:
        npz = out.with_suffix(".npz")
        # Normalize intensity to [0,1] for training paste path.
        inten_n = inten.astype(np.float32)
        if inten_n.max() > 1.5:
            inten_n = inten_n / 255.0
        np.savez_compressed(npz, xyz=xyz.astype(np.float32), intensity=inten_n)
        print(f"[ego accum] wrote {npz}")


if __name__ == "__main__":
    main()
