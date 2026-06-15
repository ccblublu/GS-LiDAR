#!/usr/bin/env python3
#
# Inference-time object removal for GS-LiDAR.
# Masks dynamic Gaussians inside OD bboxes (or all dynamic objects) during render.
#
# Examples:
#   # List tracked moving object ids
#   python scripts/render_without_object.py \
#       --config configs/gacrt_nvs_GACRT014_1740908194.yaml \
#       source_path=data \
#       model_path=eval_output/gacrt_reconstruction/GACRT014_1740908194 \
#       --list_objects
#
#   # Remove one vehicle by obj_id on test split
#   python scripts/render_without_object.py \
#       --config configs/gacrt_nvs_GACRT014_1740908194.yaml \
#       source_path=data \
#       model_path=eval_output/gacrt_reconstruction/GACRT014_1740908194 \
#       --obj_id 12 --split test
#
#   # Remove all dynamic objects
#   python scripts/render_without_object.py \
#       --config configs/gacrt_nvs_GACRT014_1740908194.yaml \
#       source_path=data \
#       model_path=eval_output/gacrt_reconstruction/GACRT014_1740908194 \
#       --remove_all_dynamic --split train
#
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import Scene, GaussianModel, RayDropPrior
from scene.gacrt_loader import (
    build_od_annotations_by_lidar_path,
    identify_moving_objects,
    load_pkl_compat,
    points_in_rotated_3d_bbox,
)
from utils.general_utils import seed_everything, visualize_depth
from utils.graphics_utils import pano_to_lidar
from utils.system_utils import save_ply

EPS = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render GS-LiDAR scenes with selected dynamic objects removed (inference only)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Checkpoint iteration (default: latest in model_path/ckpt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: model_path/render_without_object/...)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--obj_id", type=int, action="append", default=None,
                        help="Target obj_id to remove; repeatable for multiple objects")
    parser.add_argument("--remove_all_dynamic", action="store_true",
                        help="Remove all dynamic Gaussians (ignore --obj_id)")
    parser.add_argument("--bbox_scale", type=float, default=1.15,
                        help="Scale OD bbox size when masking Gaussians")
    parser.add_argument("--remove_mode", type=str, default="dynamic_in_bbox",
                        choices=["dynamic_in_bbox", "bbox"],
                        help="dynamic_in_bbox: only mask dynamic Gaussians inside bbox; "
                             "bbox: mask every Gaussian inside bbox")
    parser.add_argument("--list_objects", action="store_true",
                        help="Print moving object ids from pkl and exit")
    parser.add_argument("--no_raydrop_prior", action="store_true",
                        help="Disable RayDrop prior during rendering")
    parser.add_argument("--no_comparison", action="store_true",
                        help="Skip baseline vs removed comparison image")
    parser.add_argument("--no_ply", action="store_true",
                        help="Skip saving point cloud ply")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit number of frames to render (for quick tests)")
    parser.add_argument("--median_depth", action="store_true")
    known_args, cli_overrides = parser.parse_known_args()
    return known_args, cli_overrides


def load_merged_config(args_read, cli_overrides):
    base_conf = OmegaConf.load(args_read.base_config)
    scene_conf = OmegaConf.load(args_read.config)
    cli_conf = OmegaConf.from_cli(cli_overrides)
    conf = OmegaConf.merge(base_conf, scene_conf, cli_conf)
    OmegaConf.update(conf, "median_depth", args_read.median_depth)
    conf.shuffle = False
    conf.debug_cuda = False
    if not conf.dynamic:
        conf.t_grad = False
    return conf


def find_checkpoint(model_path, iteration=None):
    ckpt_dir = os.path.join(model_path, "ckpt")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if iteration is not None:
        path = os.path.join(ckpt_dir, f"chkpnt{iteration}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path, iteration

    iters = []
    for fname in os.listdir(ckpt_dir):
        match = re.match(r"chkpnt(\d+)\.pth", fname)
        if match:
            iters.append(int(match.group(1)))
    if not iters:
        raise FileNotFoundError(f"No chkpnt*.pth files in {ckpt_dir}")
    iteration = max(iters)
    return os.path.join(ckpt_dir, f"chkpnt{iteration}.pth"), iteration


def load_scale_factor(model_path):
    scale_path = os.path.join(model_path, "scale_factor.txt")
    if os.path.exists(scale_path):
        with open(scale_path, "r") as f:
            return float(f.read().strip())
    return 1.0


def load_transforms_frames(conf):
    json_path = os.path.join(
        conf.source_path, conf.sequence_id, f"transforms_{conf.sequence_id}_all.json"
    )
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["frames"], set(data.get("val_frame_ids", []))


def resolve_pkl_path(conf):
    pkl_path = getattr(conf, "pkl_path", None)
    if pkl_path is not None and not os.path.isabs(pkl_path):
        pkl_path = os.path.join(conf.source_path, pkl_path) if not os.path.exists(pkl_path) else pkl_path
    if pkl_path is None or not os.path.exists(pkl_path):
        candidates = glob.glob(os.path.join(conf.source_path, f"{conf.sequence_id}.pkl"))
        if candidates:
            pkl_path = candidates[0]
    if pkl_path is None or not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Cannot find pkl for sequence {conf.sequence_id}. "
            "Set pkl_path in config or place {sequence_id}.pkl under source_path."
        )
    return pkl_path


def frame_lidar_path(poses, frame_idx, conf):
    pose_info = poses[frame_idx]
    if "lidar_path" in pose_info:
        return pose_info["lidar_path"]

    pkl_path = resolve_pkl_path(conf)
    pkl_data = load_pkl_compat(pkl_path)
    pkl_idx = pose_info["idx"]
    return pkl_data[pkl_idx]["lidar_points"]["lidar_path"]


def load_od_context(conf):
    pkl_path = resolve_pkl_path(conf)
    pkl_data = load_pkl_compat(pkl_path)
    dynamic_labels = set(getattr(conf, "od_dynamic_labels", [0, 1, 2, 5, 7]))
    min_disp = getattr(conf, "od_min_displacement", 0.5)
    moving_obj_ids, obj_velocities = identify_moving_objects(pkl_data, dynamic_labels, min_disp)
    od_by_path = build_od_annotations_by_lidar_path(pkl_data, moving_obj_ids)
    return moving_obj_ids, obj_velocities, od_by_path


def scale_bbox(bbox, scale):
    return {
        "center": bbox["center"],
        "size": np.asarray(bbox["size"], dtype=np.float64) * scale,
        "yaw": bbox["yaw"],
    }


def target_bboxes_for_frame(od_by_path, lidar_path, target_obj_ids):
    annotations = od_by_path.get(lidar_path, [])
    return [bbox for obj_id, bbox in annotations if obj_id in target_obj_ids]


def build_gaussian_keep_mask(
    gaussians,
    timestamp,
    target_bboxes,
    bbox_scale,
    remove_all_dynamic,
    remove_mode,
):
    n = gaussians.get_xyz.shape[0]
    if remove_all_dynamic:
        if gaussians._is_dynamic.shape[0] == n:
            return ~gaussians.get_is_dynamic[:, 0]
        print("[Warn] is_dynamic not available; keeping all Gaussians.")
        return torch.ones(n, dtype=torch.bool, device="cuda")

    if not target_bboxes:
        return torch.ones(n, dtype=torch.bool, device="cuda")

    means_world = gaussians.get_xyz_SHM(timestamp).detach().cpu().numpy()
    scaled_bboxes = [scale_bbox(b, bbox_scale) for b in target_bboxes]
    inside_idx, _ = points_in_rotated_3d_bbox(means_world, scaled_bboxes)
    inside = torch.from_numpy(inside_idx >= 0).to(device="cuda")

    if remove_mode == "bbox":
        remove = inside
    else:
        if gaussians._is_dynamic.shape[0] != n:
            print("[Warn] is_dynamic size mismatch; falling back to bbox mode.")
            remove = inside
        else:
            remove = gaussians.get_is_dynamic[:, 0] & inside

    return ~remove



def render_view_depth(viewpoint, gaussians, args, background, env_map, gaussian_mask=None):
    render_pkg = render(
        viewpoint, gaussians, args, background,
        env_map=env_map, mask=gaussian_mask,
    )
    depth = render_pkg["depth"]
    alpha = render_pkg["alpha"]
    raydrop = render_pkg["raydrop"]

    depth_var = render_pkg["depth_square"] - depth ** 2
    depth_median = render_pkg["depth_median"]
    var_quantile = depth_var.median() * 10

    depth_mix = torch.zeros_like(depth)
    depth_mix[depth_var > var_quantile] = depth_median[depth_var > var_quantile]
    depth_mix[depth_var <= var_quantile] = depth[depth_var <= var_quantile]
    depth = torch.cat([depth_mix, depth, depth_median])

    if args.sky_depth:
        sky_depth = 900
        depth = depth / alpha.clamp_min(EPS)
        if args.depth_blend_mode == 0:
            depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
        elif args.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth

    intensity_sh = render_pkg["intensity_sh"]
    return depth, intensity_sh, raydrop


def render_range_map_masked(args, cam_front, cam_back, gaussians, background, env_map, hw, gaussian_mask=None):
    assert cam_front.towards == "forward" and cam_back.towards == "backward"
    h, w = hw
    breaks = (0, w // 2, 3 * w // 2, w * 2)

    depth_pano = torch.zeros([3, h, w * 2], device="cuda")
    intensity_sh_pano = torch.zeros([1, h, w * 2], device="cuda")
    raydrop_pano = torch.zeros([1, h, w * 2], device="cuda")
    gt_depth_pano = torch.zeros([1, h, w * 2], device="cuda")
    gt_intensity_pano = torch.zeros([1, h, w * 2], device="cuda")

    for idx, viewpoint in enumerate([cam_front, cam_back]):
        depth_gt = viewpoint.pts_depth.cuda() if viewpoint.pts_depth is not None else viewpoint.pts_depth
        intensity_gt = viewpoint.pts_intensity.cuda() if viewpoint.pts_intensity is not None else viewpoint.pts_intensity

        depth, intensity_sh, raydrop_render = render_view_depth(
            viewpoint, gaussians, args, background, env_map, gaussian_mask,
        )

        if idx % 2 == 0:
            depth_pano[:, :, breaks[1]:breaks[2]] = depth
            gt_depth_pano[:, :, breaks[1]:breaks[2]] = depth_gt
            intensity_sh_pano[:, :, breaks[1]:breaks[2]] = intensity_sh
            gt_intensity_pano[:, :, breaks[1]:breaks[2]] = intensity_gt
            raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop_render
            continue

        depth_pano[:, :, breaks[2]:breaks[3]] = depth[:, :, 0:(breaks[3] - breaks[2])]
        depth_pano[:, :, breaks[0]:breaks[1]] = depth[:, :, (w - breaks[1] + breaks[0]):w]

        gt_depth_pano[:, :, breaks[2]:breaks[3]] = depth_gt[:, :, 0:(breaks[3] - breaks[2])]
        gt_depth_pano[:, :, breaks[0]:breaks[1]] = depth_gt[:, :, (w - breaks[1] + breaks[0]):w]

        intensity_sh_pano[:, :, breaks[2]:breaks[3]] = intensity_sh[:, :, 0:(breaks[3] - breaks[2])]
        intensity_sh_pano[:, :, breaks[0]:breaks[1]] = intensity_sh[:, :, (w - breaks[1] + breaks[0]):w]

        gt_intensity_pano[:, :, breaks[2]:breaks[3]] = intensity_gt[:, :, 0:(breaks[3] - breaks[2])]
        gt_intensity_pano[:, :, breaks[0]:breaks[1]] = intensity_gt[:, :, (w - breaks[1] + breaks[0]):w]

        raydrop_pano[:, :, breaks[2]:breaks[3]] = raydrop_render[:, :, 0:(breaks[3] - breaks[2])]
        raydrop_pano[:, :, breaks[0]:breaks[1]] = raydrop_render[:, :, (w - breaks[1] + breaks[0]):w]

    return depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano


def apply_raydrop_mask(depth_pano, intensity_pano, raydrop_pano):
    raydrop_mask = torch.where(raydrop_pano > 0.5, 1, 0)
    depth_out = depth_pano * (1.0 - raydrop_mask)
    intensity_out = intensity_pano * (1.0 - raydrop_mask)
    return depth_out, intensity_out, raydrop_mask


def select_cameras(scene, conf, split, val_frame_ids):
    scale = conf.resolution_scales[-1]
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras(scale=scale)

    if split == "train":
        cameras = train_cams
    elif split == "test":
        cameras = test_cams
    else:
        cameras = train_cams + test_cams

    # Keep forward/back pairs: even indices are forward cameras.
    frame_ids = sorted({cam.colmap_id for cam in cameras if cam.towards == "forward"})
    return cameras, frame_ids


def default_output_dir(model_path, args_read):
    if args_read.remove_all_dynamic:
        tag = "all_dynamic"
    elif args_read.obj_id:
        tag = "obj_" + "_".join(str(i) for i in args_read.obj_id)
    else:
        tag = "unknown"
    return os.path.join(model_path, "render_without_object", tag)


def list_moving_objects(conf):
    moving_obj_ids, obj_velocities, _ = load_od_context(conf)
    print(f"Moving objects in {conf.sequence_id} ({len(moving_obj_ids)} total):")
    for obj_id in sorted(moving_obj_ids):
        vel = obj_velocities.get(obj_id, np.zeros(3))
        speed = float(np.linalg.norm(vel))
        print(f"  obj_id={obj_id:>6d}  |velocity|={speed:.3f} m/s  v=({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f})")


def load_model(conf, iteration=None, use_raydrop=True):
    ckpt_path, iteration = find_checkpoint(conf.model_path, iteration)
    print(f"Loading checkpoint: {ckpt_path}")

    resolution_idx = len(conf.resolution_scales) - 1
    for _ in range(iteration // conf.scale_increase_interval):
        resolution_idx = max(0, resolution_idx - 1)
    conf.resolution_scales = [conf.resolution_scales[resolution_idx]]

    gaussians = GaussianModel(conf)
    scene = Scene(conf, gaussians, shuffle=False)
    conf.scale_factor = load_scale_factor(conf.model_path)

    (model_params, _) = torch.load(ckpt_path, map_location="cuda")
    gaussians.restore(model_params, conf)

    env_map = None
    if use_raydrop:
        prior_path = ckpt_path.replace("chkpnt", "lidar_raydrop_prior_chkpnt")
        if os.path.exists(prior_path):
            start_h, start_w = scene.getWH()
            env_map = RayDropPrior(h=start_h, w=start_w).cuda()
            (prior_params, _) = torch.load(prior_path, map_location="cuda")
            env_map.restore(prior_params)
        else:
            print(f"[Warn] RayDrop prior not found: {prior_path}")

    return scene, gaussians, env_map, iteration


def camera_pair_for_frame(cameras, frame_idx, n_frames):
    cam_front = next(c for c in cameras if c.colmap_id == frame_idx and c.towards == "forward")
    cam_back = next(c for c in cameras if c.colmap_id == frame_idx + n_frames and c.towards == "backward")
    return cam_front, cam_back


def main():
    args_read, cli_overrides = parse_args()
    conf = load_merged_config(args_read, cli_overrides)
    seed_everything(conf.seed)

    if args_read.list_objects:
        if conf.scene_type != "Gacrt":
            print("--list_objects currently requires scene_type=Gacrt with pkl OD annotations.")
            return
        list_moving_objects(conf)
        return

    if not args_read.remove_all_dynamic and not args_read.obj_id:
        raise SystemExit("Specify --obj_id (repeatable) or --remove_all_dynamic.")

    if conf.scene_type != "Gacrt" and args_read.obj_id and not args_read.remove_all_dynamic:
        print("[Warn] Per-object removal via OD bbox is implemented for Gacrt. "
              "Use --remove_all_dynamic for other scene types.")

    scene, gaussians, env_map, iteration = load_model(
        conf, args_read.iteration, use_raydrop=not args_read.no_raydrop_prior,
    )

    poses, val_frame_ids = load_transforms_frames(conf)
    n_frames = len(poses)
    cameras, frame_ids = select_cameras(scene, conf, args_read.split, val_frame_ids)

    od_by_path = {}
    target_obj_ids = set(args_read.obj_id or [])
    if conf.scene_type == "Gacrt":
        moving_obj_ids, _, od_by_path = load_od_context(conf)
        unknown = target_obj_ids - moving_obj_ids
        if unknown:
            print(f"[Warn] obj_id not in moving set: {sorted(unknown)}")

    output_dir = args_read.output_dir or default_output_dir(conf.model_path, args_read)
    os.makedirs(output_dir, exist_ok=True)

    scale = conf.resolution_scales[-1]
    h, w = conf.hw[0] // scale, conf.hw[1] // scale
    bg_color = [1, 1, 1, 1] if conf.white_background else [0, 0, 0, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    meta = {
        "iteration": iteration,
        "split": args_read.split,
        "remove_all_dynamic": args_read.remove_all_dynamic,
        "obj_id": args_read.obj_id,
        "bbox_scale": args_read.bbox_scale,
        "remove_mode": args_read.remove_mode,
        "frames": [],
    }

    if args_read.max_frames is not None:
        frame_ids = frame_ids[: args_read.max_frames]

    with torch.no_grad():
        for frame_idx in tqdm(frame_ids, desc="Render without object"):
            cam_front, cam_back = camera_pair_for_frame(cameras, frame_idx, n_frames)
            lidar_path = frame_lidar_path(poses, frame_idx, conf)
            bboxes = target_bboxes_for_frame(od_by_path, lidar_path, target_obj_ids)

            keep_mask = build_gaussian_keep_mask(
                gaussians,
                cam_front.timestamp,
                bboxes,
                args_read.bbox_scale,
                args_read.remove_all_dynamic,
                args_read.remove_mode,
            )
            removed_count = int((~keep_mask).sum().item())

            depth_base, inten_base, raydrop_base, gt_depth, gt_inten = render_range_map_masked(
                conf, cam_front, cam_back, gaussians, background, env_map, [h, w], gaussian_mask=None,
            )
            depth_rm, inten_rm, raydrop_rm, _, _ = render_range_map_masked(
                conf, cam_front, cam_back, gaussians, background, env_map, [h, w], gaussian_mask=keep_mask,
            )

            depth_base, inten_base, _ = apply_raydrop_mask(depth_base[[0]], inten_base, raydrop_base)
            depth_rm, inten_rm, _ = apply_raydrop_mask(depth_rm[[0]], inten_rm, raydrop_rm)

            frame_tag = f"{frame_idx:03d}"
            if not args_read.no_ply:
                points = pano_to_lidar(depth_rm, conf.vfov, (-180, 180)) / conf.scale_factor
                save_ply(points, os.path.join(output_dir, f"{frame_tag}.ply"))

            vis_rm = visualize_depth(depth_rm, scale_factor=conf.scale_factor)
            save_image(vis_rm, os.path.join(output_dir, f"{frame_tag}_depth.png"))
            save_image(inten_rm.clamp(0, 1).repeat(3, 1, 1), os.path.join(output_dir, f"{frame_tag}_intensity.png"))

            if not args_read.no_comparison:
                vis_base = visualize_depth(depth_base, scale_factor=conf.scale_factor)
                grid = make_grid(
                    [vis_base, vis_rm,
                     gt_inten.clamp(0, 1).repeat(3, 1, 1),
                     inten_rm.clamp(0, 1).repeat(3, 1, 1)],
                    nrow=2,
                )
                save_image(grid, os.path.join(output_dir, f"{frame_tag}_compare.png"))

            meta["frames"].append({
                "frame_idx": frame_idx,
                "lidar_path": lidar_path,
                "removed_gaussians": removed_count,
                "target_bboxes": len(bboxes),
            })

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
