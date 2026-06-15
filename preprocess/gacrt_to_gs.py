"""
GACRT dataset preprocessing for GS-LiDAR.

Reads the GACRT annotation pkl file, expands all keyframes and their lidar_sweeps
into individual training frames, computes lidar2global transforms
(= ego2global since lidar2ego is identity), and saves per-frame transforms
to a JSON file compatible with the GS-LiDAR training pipeline.

By default, all sweeps (including non-keyframes) are included as training frames,
giving much denser temporal sampling (0.1s interval instead of 0.5s).

Usage:
    # Include all sweeps (249 frames)
    python preprocess/gacrt_to_gs.py --pkl data/GACRT014_1740908194.pkl

    # Keyframes only (40 frames)
    python preprocess/gacrt_to_gs.py --pkl data/GACRT014_1740908194.pkl --keyframes_only
"""

import os
import json
import argparse
import numpy as np
import pickle
from pathlib import Path


class NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that handles numpy version mismatch between save and load."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module.startswith('numpy.core'):
            module = module.replace('numpy.core', 'numpy._core')
        return super().find_class(module, name)


def main(pkl_path, frame_start=None, frame_end=None, val_interval=50,
         keyframes_only=False):
    """Process GACRT pkl file and generate transforms JSON.

    Args:
        pkl_path: Path to the GACRT annotation .pkl file.
        frame_start: Start keyframe index (inclusive). None = 0.
        frame_end: End keyframe index (inclusive). None = last keyframe.
        val_interval: Select one validation frame every this many frames.
        keyframes_only: If True, only use keyframes (40 frames). If False,
            expand all sweeps into individual training frames (249 frames).
    """
    project_root = Path(__file__).parent.parent
    pkl_path = Path(pkl_path)
    if not pkl_path.is_absolute():
        pkl_path = project_root / pkl_path

    sequence_id = pkl_path.stem

    with open(pkl_path, "rb") as f:
        data = NumpyCompatUnpickler(f).load()

    total_keyframes = len(data)
    if frame_start is None:
        frame_start = 0
    if frame_end is None:
        frame_end = total_keyframes - 1

    frame_start = max(0, frame_start)
    frame_end = min(total_keyframes - 1, frame_end)

    print(f"Total keyframes in pkl: {total_keyframes}")
    print(f"Using keyframes: {frame_start} - {frame_end}")

    if keyframes_only:
        # Only use keyframes (original behavior)
        frames_info = []
        for idx in range(frame_start, frame_end + 1):
            item = data[idx]
            lidar2global = item["ego2global"]
            timestamp = item["timestamp"]
            lidar_path = item["lidar_points"]["lidar_path"]

            frames_info.append({
                "idx": idx,
                "pkl_type": "keyframe",
                "timestamp": timestamp,
                "lidar2world": lidar2global.tolist(),
                "lidar_path": lidar_path,
                "num_pts_feats": item["lidar_points"]["num_pts_feats"],
            })
        print(f"Keyframes only: {len(frames_info)} frames")
    else:
        # Expand all sweeps into individual training frames
        # Each keyframe has 11 sweeps (5 before + self + 5 after)
        # Sweeps from adjacent keyframes overlap, so we deduplicate by lidar_path
        seen_paths = set()
        frames_info = []

        for idx in range(frame_start, frame_end + 1):
            item = data[idx]

            # Add keyframe itself
            kf_path = item["lidar_points"]["lidar_path"]
            if kf_path not in seen_paths:
                seen_paths.add(kf_path)
                frames_info.append({
                    "idx": idx,
                    "pkl_type": "keyframe",
                    "timestamp": item["timestamp"],
                    "lidar2world": item["ego2global"].tolist(),
                    "lidar_path": kf_path,
                    "num_pts_feats": item["lidar_points"]["num_pts_feats"],
                })

            # Add all sweeps
            for sweep in item["lidar_sweeps"]:
                sp = sweep["lidar_points"]
                sp_path = sp["lidar_path"]
                if sp_path not in seen_paths:
                    seen_paths.add(sp_path)
                    frames_info.append({
                        "idx": idx,
                        "pkl_type": "sweep",
                        "timestamp": sweep["timestamp"],
                        "lidar2world": sp["ego2global"].tolist(),
                        "lidar_path": sp_path,
                        "num_pts_feats": sp["num_pts_feats"],
                    })

        # Sort by timestamp for consistent ordering
        frames_info.sort(key=lambda x: x["timestamp"])

        # Re-index after sort
        for i, frame in enumerate(frames_info):
            frame["frame_idx"] = i

        print(f"All sweeps expanded: {len(frames_info)} unique frames (from {len(seen_paths)} unique .bin files)")

    # Determine validation frame indices
    total_frames = len(frames_info)
    val_frame_ids = list(range(0, total_frames, val_interval))
    # Remove first and last frames from val
    if 0 in val_frame_ids:
        val_frame_ids.remove(0)
    if (total_frames - 1) in val_frame_ids:
        val_frame_ids.remove(total_frames - 1)
    print(f"Validation frames ({len(val_frame_ids)}): {val_frame_ids[:5]}... (interval={val_interval})")

    # Panorama resolution based on Pandar128 LiDAR specs
    # 128 channels -> h=128, 0.2° horizontal -> w=900 per half
    panorama_height = 128
    panorama_width = 1800

    # Save JSON
    output_dir = pkl_path.parent / sequence_id
    output_dir.mkdir(parents=True, exist_ok=True)

    json_dict = {
        "w_lidar": panorama_width,
        "h_lidar": panorama_height,
        "aabb_scale": 2,
        "val_frame_ids": val_frame_ids,
        "frames": frames_info,
        "keyframes_only": keyframes_only,
    }

    json_path = output_dir / f"transforms_{sequence_id}_all.json"
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"Saved {json_path}")

    # Update or generate config yaml
    config_path = project_root / "configs" / f"gacrt_nvs_{sequence_id}.yaml"
    total_duration = frames_info[-1]["timestamp"] - frames_info[0]["timestamp"]

    config_content = f"""# GACRT config for sequence {sequence_id}
# Pandar128: 128 channels, vfov=[-25,15], hfov=[-180,180], horizontal 0.2° resolution
# Dynamic scene config (moving vehicles, pedestrians, etc.)
# {'Keyframes only (40 frames)' if keyframes_only else f'All sweeps expanded ({total_frames} frames, 0.1s interval)'}
# Auto-generated by preprocess/gacrt_to_gs.py

exhaust_test: false

# ModelParams
scene_type: "Gacrt"
resolution_scales: [ 1, 2, 4 ]
scale_increase_interval: 5000
cam_num: 2
eval: True
separate_scaling_t: 0.2

sequence_id: "{sequence_id}"
pkl_path: "data/{sequence_id}.pkl"
keyframes_only: {'true' if keyframes_only else 'false'}
vfov: [ -25.0, 15.0 ]
hfov: [ -90, 90 ]
hw: [ 128, 900 ]


# OptimizationParams
iterations: 30000

densify_until_iter: 15000
densify_grad_threshold: 1e-4
sh_increase_interval: 2000

lambda_lidar: 1e1
lambda_lidar_median: 1e1
lidar_decay: 0

only_velodyne: True
lambda_intensity: 0.05
lambda_intensity_sh: 0.05
lambda_raydrop: 0.05
lambda_smooth: 1e-3
lambda_chamfer: 1e-1
lambda_distortion: 1e-2
lambda_normal_consistency: 1e-1
thresh_opa_prune: 0.1

test_iterations: [ 3000, 7000, 15000, 30000 ]
random_init_point: 0
dynamic: True
sky_depth: False

opacity_lr: 0.05
velocity_lr: 0.001
time_split_frac: 1.0
lambda_self_supervision: 0.5
lambda_v_reg: 1e0
t_init: 0.006

num_pts: 3000000
densify_until_num_points: 10000000
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"{'Updated' if config_path.exists() else 'Generated'} config: {config_path}")

    print(f"\nDone! {total_frames} training frames, duration={total_duration:.1f}s")
    print(f"Frame interval: ~{total_duration/max(total_frames-1,1):.2f}s")
    print(f"\nRun training:")
    print(f"  CUDA_VISIBLE_DEVICES=0 python train.py \\")
    print(f"    --config configs/gacrt_nvs_{sequence_id}.yaml \\")
    print(f"    source_path=data \\")
    print(f"    model_path=eval_output/gacrt_reconstruction/{sequence_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GACRT dataset preprocessing for GS-LiDAR")
    parser.add_argument("--pkl", type=str, required=True, help="Path to GACRT annotation .pkl file")
    parser.add_argument("--frame_start", type=int, default=None, help="Start keyframe index (inclusive)")
    parser.add_argument("--frame_end", type=int, default=None, help="End keyframe index (inclusive)")
    parser.add_argument("--val_interval", type=int, default=50, help="Validation frame interval")
    parser.add_argument("--keyframes_only", action="store_true", help="Only use keyframes, skip sweeps")
    args = parser.parse_args()
    main(args.pkl, args.frame_start, args.frame_end, args.val_interval, args.keyframes_only)
