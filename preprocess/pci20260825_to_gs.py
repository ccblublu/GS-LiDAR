"""Prepare PCI_20260825 fused SLAM lidar for GS-LiDAR.

The share-data slam folder stores top (OT128) + 左右补盲 (QT128) fused clouds
(source channel) and IMU-frame poses. This script writes a transforms JSON and
symlinks PCDs under data/ without copying the original files.

Usage:
    python preprocess/pci20260825_to_gs.py
    python preprocess/pci20260825_to_gs.py --stride 5
    python preprocess/pci20260825_to_gs.py --tracks 0 --stride 2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

SRC_ROOT = Path("/mnt/share_data/chenping/fy_parking_3dgrt/PCI_20260825/slam")
SEQUENCE_ID = "PCI_20260825"
# Fused PCD source channel: 0=OT128 top, 1=left, 2=right, 3=front, 4=back QT128
INPUT_LIDAR_SOURCES = [0, 1, 2, 3, 4]


def flatten_frames(tracks, selected_tracks=None):
    frames = []
    for track in tracks:
        if selected_tracks is not None and track["track_id"] not in selected_tracks:
            continue
        for fr in track["frames"]:
            item = dict(fr)
            item["track_id"] = track["track_id"]
            item["segment"] = fr.get("segment", track.get("segment"))
            frames.append(item)
    frames.sort(key=lambda x: (x["timestamp_ms"], x["track_id"], x["frame_id"]))
    return frames


def main():
    parser = argparse.ArgumentParser(description="PCI_20260825 SLAM → GS-LiDAR transforms")
    parser.add_argument("--src_root", type=str, default=str(SRC_ROOT))
    parser.add_argument("--tracks", type=int, nargs="*", default=None,
                        help="Keep only these track_id values (default: all)")
    parser.add_argument("--stride", type=int, default=5, help="Keep every Nth frame after sort")
    parser.add_argument("--val_interval", type=int, default=50)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    json_src = src_root / "frame_list.json"
    pcd_dir = src_root / "pcd"
    if not json_src.is_file():
        raise FileNotFoundError(json_src)
    if not pcd_dir.is_dir():
        raise FileNotFoundError(pcd_dir)

    tracks = json.loads(json_src.read_text())
    selected = set(args.tracks) if args.tracks else None
    raw_frames = flatten_frames(tracks, selected)
    n_all = len(raw_frames)
    if args.stride > 1:
        raw_frames = raw_frames[:: args.stride]
    if not raw_frames:
        raise RuntimeError("No frames selected")

    project_root = Path(__file__).resolve().parent.parent
    json_dir = project_root / "data" / SEQUENCE_ID
    lidar_dir = project_root / "data" / "raw_data" / SEQUENCE_ID / "lidar"
    json_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    missing = []
    used = set()
    for i, fr in enumerate(raw_frames):
        pcd_name = Path(fr["lidars"]["slam"]).name
        src = pcd_dir / pcd_name
        if not src.is_file():
            missing.append(str(src))
            continue
        if pcd_name in used:
            raise RuntimeError(f"Duplicate PCD match: {pcd_name}")
        used.add(pcd_name)

        dest = lidar_dir / pcd_name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        os.symlink(src.resolve(), dest)

        T = np.asarray(fr["pose"], dtype=np.float64)
        det = np.linalg.det(T[:3, :3])
        if abs(abs(det) - 1.0) > 0.05:
            raise RuntimeError(f"Non-rigid pose at {i}: det={det}")

        frames.append({
            "idx": i,
            "pkl_type": "keyframe",
            "timestamp": fr["timestamp_ms"] / 1000.0,
            "lidar2world": T.tolist(),
            "lidar_path": f"raw_data/{SEQUENCE_ID}/lidar/{pcd_name}",
            "num_pts_feats": 6,
            "frame_idx": len(frames),
            "track_id": fr["track_id"],
            "segment": fr.get("segment"),
            "pcd_name": pcd_name,
        })

    if missing:
        print(f"[warn] missing {len(missing)} pcds, e.g. {missing[0]}")
    if not frames:
        raise RuntimeError("No pose/PCD pairs matched")

    total = len(frames)
    val_frame_ids = list(range(0, total, args.val_interval))
    if 0 in val_frame_ids:
        val_frame_ids.remove(0)
    if (total - 1) in val_frame_ids:
        val_frame_ids.remove(total - 1)

    json_dict = {
        "w_lidar": 1800,
        "h_lidar": 128,
        "aabb_scale": 2,
        "val_frame_ids": val_frame_ids,
        "frames": frames,
        "keyframes_only": False,
        "pose_frame": "imu",
        "point_frame": "imu",
        "lidar_source": "slam_fused_top_blind",
        "input_lidar_source": INPUT_LIDAR_SOURCES,
        "source_channel": {
            "0": "top_OT128",
            "1": "blind_QT128_left",
            "2": "blind_QT128_right",
            "3": "blind_QT128_front",
            "4": "blind_QT128_back",
        },
        "lidar_models": {
            "top": "Hesai OT128",
            "blind": "Hesai QT128",
        },
        "frame_stride": args.stride,
    }
    json_path = json_dir / f"transforms_{SEQUENCE_ID}_all.json"
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)

    stale = 0
    for p in lidar_dir.glob("*.pcd"):
        if p.name not in used:
            p.unlink()
            stale += 1

    print(f"Kept {len(frames)}/{n_all} frames (stride={args.stride}) from {json_src}")
    print(f"tracks={sorted({fr['track_id'] for fr in frames})}")
    print(f"val_frame_ids: {len(val_frame_ids)} -> {val_frame_ids[:8]}...")
    print(f"Wrote {json_path}")
    print(f"Symlinked {len(frames)} files under {lidar_dir}"
          + (f", removed {stale} unused links" if stale else ""))


if __name__ == "__main__":
    main()
