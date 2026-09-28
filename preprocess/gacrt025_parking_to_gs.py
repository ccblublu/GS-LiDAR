"""Prepare GACRT025 parking clip for GS-LiDAR without copying raw clouds.

Reads SLAM ego poses (vehicle-body → world) and symlinks lidar_demotion/top
PCD files under data/. Does not modify the original share-data directory.

Usage:
    python preprocess/gacrt025_parking_to_gs.py
"""

import json
import os
from pathlib import Path

import numpy as np

SRC_ROOT = Path("/mnt/share_data/FY_data/parking/GACRT025_1757941510")
SLAM_POSE = Path("/mnt/share_data/FY_data/parking/slam/GACRT025_1757941510_pose.txt")
DEMOTION_TOP = SRC_ROOT / "lidar_demotion" / "top"
SEQUENCE_ID = "GACRT025_1757941510"
MAX_STAMP_DT_NS = 20_000_000  # 20 ms


def quat_to_rot(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_to_mat(x, y, z, qx, qy, qz, qw):
    T = np.eye(4, dtype=np.float64)
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)
    T[:3, :3] = quat_to_rot(*q)
    T[:3, 3] = [x, y, z]
    return T


def load_slam_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            t, x, y, z, qx, qy, qz, qw = map(float, parts[:8])
            stamp = int(float(parts[8]))
            poses.append({
                "t": t,
                "stamp": stamp,
                "T": pose_to_mat(x, y, z, qx, qy, qz, qw),
            })
    return poses


def index_demotion_pcds(folder):
    items = []
    for p in sorted(folder.glob("*.pcd")):
        stamp = int(p.name.split("_")[0])
        items.append((stamp, p))
    return items


def match_pose_to_pcd(stamp, pcds):
    best = None
    best_dt = None
    for pcd_stamp, path in pcds:
        dt = abs(pcd_stamp - stamp)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = (pcd_stamp, path)
    if best is None or best_dt > MAX_STAMP_DT_NS:
        return None, best_dt
    return best, best_dt


def main():
    project_root = Path(__file__).resolve().parent.parent
    json_dir = project_root / "data" / SEQUENCE_ID
    lidar_dir = project_root / "data" / "raw_data" / SEQUENCE_ID / "lidar"
    json_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    poses = load_slam_poses(SLAM_POSE)
    pcds = index_demotion_pcds(DEMOTION_TOP)
    if not poses:
        raise RuntimeError(f"No SLAM poses in {SLAM_POSE}")
    if not pcds:
        raise RuntimeError(f"No demotion PCDs in {DEMOTION_TOP}")

    frames = []
    used_targets = set()
    for i, pose in enumerate(poses):
        match, dt = match_pose_to_pcd(pose["stamp"], pcds)
        if match is None:
            print(f"[skip] pose stamp={pose['stamp']} dt={dt}")
            continue
        _, src = match
        if src.name in used_targets:
            raise RuntimeError(f"Duplicate PCD match: {src.name}")
        used_targets.add(src.name)

        dest = lidar_dir / src.name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        os.symlink(src.resolve(), dest)

        rel_path = f"raw_data/{SEQUENCE_ID}/lidar/{src.name}"
        T = pose["T"]
        det = np.linalg.det(T[:3, :3])
        if abs(abs(det) - 1.0) > 0.05:
            raise RuntimeError(f"Non-rigid pose at {i}: det={det}")

        frames.append({
            "idx": i,
            "pkl_type": "keyframe",
            "timestamp": pose["t"],
            "lidar2world": T.tolist(),
            "lidar_path": rel_path,
            "num_pts_feats": 6,
            "frame_idx": len(frames),
            "slam_stamp": pose["stamp"],
            "pcd_name": src.name,
            "stamp_dt_ns": int(dt),
        })

    if not frames:
        raise RuntimeError("No pose/PCD pairs matched")

    total = len(frames)
    val_frame_ids = list(range(0, total, 50))
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
        "pose_frame": "ego",
        "point_frame": "ego",
        "lidar_source": "lidar_demotion/top",
    }
    json_path = json_dir / f"transforms_{SEQUENCE_ID}_all.json"
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)

    dts = [fr["stamp_dt_ns"] for fr in frames]
    print(f"Matched {len(frames)}/{len(poses)} SLAM poses to demotion PCDs")
    print(f"stamp dt ns: min={min(dts)} med={int(np.median(dts))} max={max(dts)}")
    print(f"val_frame_ids: {val_frame_ids}")
    print(f"Wrote {json_path}")
    print(f"Symlinked {len(frames)} files under {lidar_dir}")


if __name__ == "__main__":
    main()
