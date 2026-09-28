#!/usr/bin/env python3
#
# Hybrid render: GS-LiDAR static background + XSIM dynamic layer (from cache).
#
# Dynamic layer is read from pre-rendered numpy cache (see --dynamic_cache).
# Use XSIM tools/render.py to produce cache, or workspace/tools/apply_edit_spec.py.
#
# Example:
#   python scripts/render_hybrid.py \
#       --config configs/gacrt_static_bg.yaml \
#       source_path=data \
#       model_path=eval_output/hybrid/GACRT014_static_bg \
#       --dynamic_cache ../dynamic_cache/GACRT014 \
#       --split test
#
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_without_object import (
    apply_raydrop_mask,
    iter_frame_sensor_pairs,
    list_sensor_ids,
    load_merged_config,
    load_model,
    load_transforms_frames,
    render_range_map_masked,
    select_cameras,
    sensor_name,
)
from utils.general_utils import seed_everything, visualize_depth
from utils.graphics_utils import pano_to_lidar
from utils.static_confidence import StaticConfidenceTracker
from utils.system_utils import save_ply

EPS = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid GS-LiDAR static + XSIM dynamic render.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--dynamic_cache", type=str, required=True,
                        help="Directory with per-frame dynamic layer npz (depth/intensity/alpha pano)")
    parser.add_argument("--alpha_thresh", type=float, default=0.5)
    parser.add_argument("--hole_policy", type=str, default="keep",
                        choices=["keep", "raydrop"],
                        help="Policy for low-confidence holes after compose")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--sensor_id", type=int, action="append", default=None,
        help="Blind-lidar slots to render (0=left…3=back). Default: all.",
    )
    parser.add_argument("--no_ply", action="store_true")
    parser.add_argument("--no_raydrop_prior", action="store_true")
    parser.add_argument("--save_comparison", action="store_true")
    parser.add_argument("--median_depth", action="store_true")
    known, cli = parser.parse_known_args()
    return known, cli


def load_dynamic_cache(cache_dir: str, frame_idx: int):
    """Load dynamic layer pano from cache/frame_XXX.npz."""
    path = os.path.join(cache_dir, f"frame_{frame_idx:03d}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    depth = torch.from_numpy(data["depth"]).float().cuda()
    intensity = torch.from_numpy(data["intensity"]).float().cuda()
    alpha = torch.from_numpy(data["alpha"]).float().cuda() if "alpha" in data else (depth > EPS).float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if intensity.ndim == 2:
        intensity = intensity.unsqueeze(0)
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    return depth, intensity, alpha


def compose_hybrid(depth_s, inten_s, raydrop_s, depth_d, inten_d, alpha_d, alpha_thresh=0.5):
    """Min-depth compose: dynamic wins when closer and alpha sufficient."""
    mask_d = (depth_d > EPS) & (alpha_d > alpha_thresh)
    use_d = mask_d & ((depth_s <= EPS) | (depth_d < depth_s))
    depth_out = torch.where(use_d, depth_d, depth_s)
    inten_out = torch.where(use_d, inten_d, inten_s)
    return depth_out, inten_out, use_d


def load_confidence(model_path, scale, towards):
    conf_path = os.path.join(model_path, "static_confidence.npz")
    if not os.path.exists(conf_path):
        return None
    data = np.load(conf_path)
    key = f"{towards}_{scale}"
    if key not in data:
        return None
    tracker = StaticConfidenceTracker(data[key].shape[0], data[key].shape[1])
    tracker.counts = data[key].astype(np.int32)
    return tracker


def apply_hole_policy(depth, intensity, model_path, scale, hole_policy):
    if hole_policy != "raydrop":
        return depth, intensity
    conf_fwd = load_confidence(model_path, scale, "forward")
    conf_bck = load_confidence(model_path, scale, "backward")
    if conf_fwd is None and conf_bck is None:
        return depth, intensity
    h, w2 = depth.shape[-2], depth.shape[-1]
    w = w2 // 2
    hole = depth <= EPS
    if conf_fwd is not None:
        low = hole[:, :, :w] & (~torch.from_numpy(conf_fwd.query_mask()).to(depth.device))
        depth[:, :, :w][low] = 0
        intensity[:, :, :w][low] = 0
    if conf_bck is not None:
        low = hole[:, :, w:] & (~torch.from_numpy(conf_bck.query_mask()).to(depth.device))
        depth[:, :, w:][low] = 0
        intensity[:, :, w:][low] = 0
    return depth, intensity


def main():
    args_read, cli = parse_args()
    conf = load_merged_config(args_read, cli)
    seed_everything(conf.seed)

    scene, gaussians, env_map, iteration = load_model(
        conf, args_read.iteration, use_raydrop=not args_read.no_raydrop_prior,
    )
    poses, val_frame_ids = load_transforms_frames(conf)
    n_frames = len(poses)
    cameras, frame_ids = select_cameras(scene, conf, args_read.split, val_frame_ids)

    output_dir = args_read.output_dir or os.path.join(
        conf.model_path, "render_hybrid", os.path.basename(args_read.dynamic_cache.rstrip("/"))
    )
    os.makedirs(output_dir, exist_ok=True)

    scale = conf.resolution_scales[-1]
    h, w = conf.hw[0] // scale, conf.hw[1] // scale
    bg = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device="cuda")

    if args_read.max_frames is not None:
        frame_ids = frame_ids[: args_read.max_frames]

    sensor_ids = (
        sorted(set(args_read.sensor_id)) if args_read.sensor_id
        else list_sensor_ids(cameras)
    )
    multi_sensor = len(sensor_ids) > 1
    pairs = list(iter_frame_sensor_pairs(cameras, frame_ids, n_frames, sensor_ids))
    if not pairs:
        raise SystemExit(
            f"No camera pairs for frame_ids={frame_ids} sensor_ids={sensor_ids}"
        )

    meta = {
        "iteration": iteration,
        "dynamic_cache": args_read.dynamic_cache,
        "sensor_ids": sensor_ids,
        "frames": [],
    }

    with torch.no_grad():
        for frame_idx, sid, cam_front, cam_back in tqdm(pairs, desc="Hybrid render"):
            sname = sensor_name(sid)
            tag = f"{frame_idx:03d}_{sname}" if multi_sensor else f"{frame_idx:03d}"

            depth_s, inten_s, raydrop_s, gt_depth, gt_inten = render_range_map_masked(
                conf, cam_front, cam_back, gaussians, bg, env_map, [h, w], gaussian_mask=None,
            )
            depth_s = depth_s[[0]]
            inten_s = inten_s

            dyn = load_dynamic_cache(args_read.dynamic_cache, frame_idx)
            if dyn is None:
                depth_out, inten_out = depth_s, inten_s
                used_dynamic = False
            else:
                depth_d, inten_d, alpha_d = dyn
                if depth_d.shape[-2:] != depth_s.shape[-2:]:
                    depth_d = torch.nn.functional.interpolate(
                        depth_d.unsqueeze(0), size=depth_s.shape[-2:], mode="bilinear", align_corners=False
                    ).squeeze(0)
                    inten_d = torch.nn.functional.interpolate(
                        inten_d.unsqueeze(0), size=inten_s.shape[-2:], mode="bilinear", align_corners=False
                    ).squeeze(0)
                    alpha_d = torch.nn.functional.interpolate(
                        alpha_d.unsqueeze(0), size=depth_s.shape[-2:], mode="nearest"
                    ).squeeze(0)
                depth_out, inten_out, _ = compose_hybrid(
                    depth_s, inten_s, raydrop_s, depth_d, inten_d, alpha_d, args_read.alpha_thresh,
                )
                used_dynamic = True

            depth_out, inten_out, _ = apply_raydrop_mask(depth_out, inten_out, raydrop_s)
            depth_out, inten_out = apply_hole_policy(
                depth_out, inten_out, conf.model_path, scale, args_read.hole_policy,
            )

            if not args_read.no_ply:
                pts = pano_to_lidar(depth_out, conf.vfov, (-180, 180)) / conf.scale_factor
                save_ply(pts, os.path.join(output_dir, f"{tag}.ply"))

            save_image(
                visualize_depth(depth_out, scale_factor=conf.scale_factor),
                os.path.join(output_dir, f"{tag}_depth.png"),
            )
            save_image(
                inten_out.clamp(0, 1).repeat(3, 1, 1),
                os.path.join(output_dir, f"{tag}_intensity.png"),
            )

            if args_read.save_comparison and dyn is not None:
                vis_s = visualize_depth(depth_s, scale_factor=conf.scale_factor)
                vis_h = visualize_depth(depth_out, scale_factor=conf.scale_factor)
                save_image(torch.cat([vis_s, vis_h], dim=2), os.path.join(output_dir, f"{tag}_compare.png"))

            meta["frames"].append({
                "frame_idx": frame_idx,
                "sensor_id": sid,
                "sensor": sname,
                "used_dynamic": used_dynamic,
            })

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Done. {len(pairs)} renders "
        f"({len(frame_ids)} frames × {len(sensor_ids)} sensors) → {output_dir}"
    )


if __name__ == "__main__":
    main()
