#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import sys
import json
import time
import os
import shutil
from collections import defaultdict
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import psnr, ssim, inverse_depth_smoothness_loss_mask, tv_loss
from gaussian_renderer import render, render_range_map
from scene import Scene, GaussianModel, RayDropPrior
from utils.general_utils import seed_everything, visualize_depth
from utils.graphics_utils import pano_to_lidar
from utils.system_utils import save_ply
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
import numpy as np
from omegaconf import OmegaConf
from utils.graphics_utils import depth_to_normal
from utils.metrics_utils import DepthMeter, PointsMeter, RaydropMeter, IntensityMeter
from utils.training_logger import TrainingLogger
from utils.static_confidence import StaticConfidenceTracker
from utils.od_loss_mask import resize_od_mask, stitch_od_mask_pano
from utils.blind_dead_mask import dead_mask_enabled, get_camera_mask, stitch_camera_mask, force_dead_drop
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from scene.unet import UNet

EPS = 1e-5


def get_od_valid_mask(viewpoint_cam, ref_map, args):
    """True outside OD silhouette (supervised); False inside OD (unsupervised)."""
    if not getattr(args, 'od_mask_gt_loss', False):
        return torch.ones_like(ref_map, dtype=torch.bool)
    od_mask = getattr(viewpoint_cam, 'od_loss_mask', None)
    if od_mask is None:
        return torch.ones_like(ref_map, dtype=torch.bool)
    od_mask = resize_od_mask(od_mask, ref_map.shape[-2], ref_map.shape[-1]).to(ref_map.device)
    return od_mask > 0.5


def get_supervision_mask(viewpoint_cam, pts_depth, args):
    """Valid pixels for depth/intensity loss (excludes OD regions when od_mask_gt_loss)."""
    mask = pts_depth > 0
    if getattr(args, 'od_mask_gt_loss', False):
        mask = mask & get_od_valid_mask(viewpoint_cam, pts_depth, args)
    return mask


def get_raydrop_gt(viewpoint_cam):
    """Raydrop GT from the training-input sensor (top-only when pts_depth_input exists)."""
    depth_in = getattr(viewpoint_cam, "pts_depth_input", None)
    if depth_in is None:
        depth_in = viewpoint_cam.pts_depth
    return 1.0 - (depth_in.cuda() > 0).float()


def get_raydrop_valid_mask(viewpoint_cam, ref_map, args):
    """Pixels that participate in raydrop loss. Body hits and the blind dead zone are ignore."""
    valid = get_od_valid_mask(viewpoint_cam, ref_map, args)
    ego = getattr(viewpoint_cam, "ego_body_mask", None)
    if ego is not None:
        ego = resize_od_mask(ego, ref_map.shape[-2], ref_map.shape[-1]).to(ref_map.device)
        valid = valid & (ego <= 0.5)
    dead_ignore = get_camera_mask(
        viewpoint_cam, ref_map.shape[-2], ref_map.shape[-1], args, "blind_ignore_mask")
    if dead_ignore is not None:
        valid = valid & (dead_ignore.to(ref_map.device) <= 0.5)
    return valid


def stitch_camera_depth(cam_front, cam_back, hw, attr="pts_depth"):
    """Camera depth attr: HxW if ``cam_back is None``, else forward/back stitch Hx(2W)."""
    h, w = hw
    front = getattr(cam_front, attr, None)
    if cam_back is None:
        if front is None:
            return None
        front = front.cuda()
        if front.shape[-2:] != (h, w):
            front = F.interpolate(front.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
        return front
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    back = getattr(cam_back, attr, None)
    if front is None or back is None:
        return None
    front = front.cuda()
    back = back.cuda()
    if front.shape[-2:] != (h, w):
        front = F.interpolate(front.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
        back = F.interpolate(back.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    out = torch.zeros([1, h, w * 2], device=front.device)
    out[:, :, breaks[1]:breaks[2]] = front
    out[:, :, breaks[2]:breaks[3]] = back[:, :, 0:(breaks[3] - breaks[2])]
    out[:, :, breaks[0]:breaks[1]] = back[:, :, (w - breaks[1] + breaks[0]):w]
    return out


def compute_hole_stats(pred_depth, gt_depth, od_valid=None):
    """Static-miss / pred-empty / OD-empty rates for one pano."""
    pred_hit = pred_depth > 0
    gt_hit = gt_depth > 0
    static_miss = (gt_hit & ~pred_hit).float().sum() / gt_hit.float().sum().clamp_min(1)
    pred_empty = (~pred_hit).float().mean()
    stats = {
        "static_miss": float(static_miss.item()),
        "pred_empty": float(pred_empty.item()),
    }
    if od_valid is not None:
        od_region = ~od_valid
        od_empty = (od_region & ~pred_hit).float().sum() / od_region.float().sum().clamp_min(1)
        stats["od_empty"] = float(od_empty.item())
        stats["od_frac"] = float(od_region.float().mean().item())
    return stats


# training_logger = None


def training(args):
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)

    gaussians = GaussianModel(args)

    scene = Scene(args, gaussians, shuffle=args.shuffle)
    with open(os.path.join(args.model_path, 'scale_factor.txt'), 'w') as f:
        f.writelines(str(args.scale_factor))

    gaussians.training_setup(args)

    start_w, start_h = scene.getWH()
    n_sensors = int(getattr(args, "n_lidar_sensors", 1) or 1)
    lidar_raydrop_prior = RayDropPrior(h=start_h, w=start_w, n_sensors=n_sensors).cuda()
    lidar_raydrop_prior.training_setup(args)

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)

        lidar_raydrop_prior_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                                      os.path.basename(args.start_checkpoint).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
        (lidar_raydrop_prior_params, _) = torch.load(lidar_raydrop_prior_checkpoint)
        lidar_raydrop_prior.restore(lidar_raydrop_prior_params, args)

        for i in range(first_iter // args.scale_increase_interval):
            scene.upScale()

    bg_color = [1, 1, 1, 1] if args.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.test_only or first_iter == args.iterations:
        with torch.no_grad():
            complete_eval(first_iter, args.test_iterations, scene, render, (args, background),
                          {}, env_map=lidar_raydrop_prior)
            return

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    conf_trackers = {}
    if getattr(args, 'static_confidence_track', False):
        for res_scale in scene.resolution_scales:
            w, h = scene.wh[res_scale]
            conf_trackers[res_scale] = {
                'forward': StaticConfidenceTracker(h, w),
                'backward': StaticConfidenceTracker(h, w),
            }

    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress", miniters=10)

    for iteration in progress_bar:
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % args.sh_increase_interval == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]

        # render v and t scale map
        v = gaussians.get_inst_velocity
        t_scale = gaussians.get_scaling_t.clamp_max(2)
        other = [t_scale, v]

        if np.random.random() < args.lambda_self_supervision:
            time_shift = 3 * (np.random.random() - 0.5) * scene.time_interval
        else:
            time_shift = None

        render_pkg = render(viewpoint_cam, gaussians, args, background, env_map=lidar_raydrop_prior, other=other, time_shift=time_shift, is_training=True)

        depth = render_pkg["depth"]
        depth_median = render_pkg["depth_median"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        log_dict = {}

        feature = render_pkg['feature'] / alpha.clamp_min(EPS)
        t_map = feature[0:1]
        v_map = feature[1:4]

        intensity_sh_map = render_pkg['intensity_sh']
        raydrop_map = render_pkg['raydrop']

        if args.sky_depth:
            sky_depth = 900
            depth = depth / alpha.clamp_min(EPS)
            if args.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        loss = 0
        if args.lambda_distortion > 0:
            lambda_dist = args.lambda_distortion if iteration > 3000 else 0.0
            distortion = render_pkg["distortion"]
            loss_distortion = distortion.mean()
            log_dict['loss_distortion'] = loss_distortion.item()
            loss += lambda_dist * loss_distortion

        if args.lambda_lidar > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = get_supervision_mask(viewpoint_cam, pts_depth, args)
            loss_lidar = F.l1_loss(pts_depth[mask], depth[mask])
            if args.lidar_decay > 0:
                iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
            else:
                iter_decay = 1
            log_dict['loss_lidar'] = loss_lidar.item()
            loss += iter_decay * args.lambda_lidar * loss_lidar

        if args.lambda_lidar_median > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = get_supervision_mask(viewpoint_cam, pts_depth, args)
            loss_lidar_median = F.l1_loss(pts_depth[mask], depth_median[mask])
            log_dict['loss_lidar_median'] = loss_lidar_median.item()
            loss += args.lambda_lidar_median * loss_lidar_median

        if args.lambda_t_reg > 0:
            loss_t_reg = -torch.abs(t_map).mean()
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

        if args.lambda_v_reg > 0:
            # OD-aware velocity regularization: differentiate static vs dynamic
            velocity = gaussians.get_inst_velocity  # (N, 3)
            is_dynamic = gaussians.get_is_dynamic  # (N, 1) bool

            lambda_v_static = getattr(args, 'lambda_v_reg_static', args.lambda_v_reg)
            lambda_v_dynamic = getattr(args, 'lambda_v_reg_dynamic', args.lambda_v_reg)

            if is_dynamic.any():
                static_mask = ~is_dynamic[:, 0]
                dynamic_mask = is_dynamic[:, 0]

                # Static: strong regularization, push velocity to zero
                if static_mask.any():
                    loss_v_static = torch.abs(velocity[static_mask]).mean()
                else:
                    loss_v_static = torch.tensor(0.0, device="cuda")

                # Dynamic: weak regularization, allow free motion
                if dynamic_mask.any():
                    loss_v_dynamic = torch.abs(velocity[dynamic_mask]).mean()
                else:
                    loss_v_dynamic = torch.tensor(0.0, device="cuda")

                loss_v_reg = lambda_v_static * loss_v_static + lambda_v_dynamic * loss_v_dynamic
                log_dict['loss_v_static'] = loss_v_static.item()
                log_dict['loss_v_dynamic'] = loss_v_dynamic.item()
            else:
                # No OD labels, use uniform regularization on rendered v_map
                loss_v_reg = torch.abs(v_map).mean()

            log_dict['loss_v_reg'] = loss_v_reg.item()
            loss += loss_v_reg

        # Intensity sh
        if args.lambda_intensity_sh > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = get_supervision_mask(viewpoint_cam, pts_depth, args)
            pts_intensity = viewpoint_cam.pts_intensity.cuda()
            loss_intensity_sh = torch.nn.functional.l1_loss(pts_intensity[mask], intensity_sh_map[mask])
            log_dict['loss_intensity_sh'] = loss_intensity_sh.item()
            loss += args.lambda_intensity_sh * loss_intensity_sh

        if args.lambda_raydrop > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            # Hits = training-input returns (top-only when pts_depth_input is set).
            # OD silhouette and ego-body pixels are unsupervised, not "must drop".
            gt_raydrop = get_raydrop_gt(viewpoint_cam)
            raydrop_valid = get_raydrop_valid_mask(viewpoint_cam, pts_depth, args)
            if raydrop_valid.any():
                loss_raydrop = F.binary_cross_entropy(raydrop_map[raydrop_valid], gt_raydrop[raydrop_valid])
            else:
                loss_raydrop = torch.zeros((), device=raydrop_map.device)
            log_dict['loss_raydrop'] = loss_raydrop.item()
            loss += args.lambda_raydrop * loss_raydrop

        # chamfer loss
        if args.lambda_chamfer > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = get_supervision_mask(viewpoint_cam, pts_depth, args).float()

            cham_fn = chamfer_3DDist()
            pred_lidar = pano_to_lidar(depth * mask, args.vfov, args.hfov) / args.scale_factor
            gt_lidar = pano_to_lidar(pts_depth * mask, args.vfov, args.hfov) / args.scale_factor
            if pred_lidar.numel() == 0 or gt_lidar.numel() == 0:
                loss_chamfer = torch.tensor(0.0, device=depth.device)
            else:
                dist1, dist2, _, _ = cham_fn(pred_lidar[None], gt_lidar[None])
                loss_chamfer = dist1.mean() + dist2.mean()
            log_dict['loss_chamfer'] = loss_chamfer.item()
            loss += args.lambda_chamfer * loss_chamfer

        if args.lambda_smooth > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            od_valid = get_od_valid_mask(viewpoint_cam, pts_depth, args).float()
            gt_grad_x = pts_depth[:, :, :-1] - pts_depth[:, :, 1:]
            gt_grad_y = pts_depth[:, :-1, :] - pts_depth[:, 1:, :]
            mask_x = (torch.where(pts_depth[:, :, :-1] > 0, 1, 0) *
                      torch.where(pts_depth[:, :, 1:] > 0, 1, 0) *
                      torch.where(od_valid[:, :, :-1] > 0.5, 1, 0) *
                      torch.where(od_valid[:, :, 1:] > 0.5, 1, 0))
            mask_y = (torch.where(pts_depth[:, :-1, :] > 0, 1, 0) *
                      torch.where(pts_depth[:, 1:, :] > 0, 1, 0) *
                      torch.where(od_valid[:, :-1, :] > 0.5, 1, 0) *
                      torch.where(od_valid[:, 1:, :] > 0.5, 1, 0))

            grad_clip = 0.01 * args.scale_factor
            grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip, 1, 0) * mask_x
            grad_mask_x = grad_mask_x.bool()
            grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip, 1, 0) * mask_y
            grad_mask_y = grad_mask_y.bool()

            pred_grad_x = depth[:, :, :-1] - depth[:, :, 1:]
            pred_grad_y = depth[:, :-1, :] - depth[:, 1:, :]
            if grad_mask_x.any() or grad_mask_y.any():
                loss_smooth = torch.tensor(0.0, device=depth.device)
                if grad_mask_x.any():
                    loss_smooth = loss_smooth + F.l1_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x])
                if grad_mask_y.any():
                    loss_smooth = loss_smooth + F.l1_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y])
            else:
                loss_smooth = torch.tensor(0.0, device=depth.device)
            log_dict['loss_smooth'] = loss_smooth.item()
            loss += args.lambda_smooth * loss_smooth

        if args.lambda_tv > 0:
            loss_tv = tv_loss(depth)
            log_dict['loss_tv'] = loss_tv.item()
            loss += args.lambda_tv * loss_tv

        # 每个gaussian的opa 而不是render的 没用
        if args.lambda_gs_opa > 0:
            o = gaussians.get_opacity.clamp(1e-6, 1 - 1e-6)
            loss_gs_opa = ((1 - o) ** 2).mean()
            log_dict['loss_depth_opa'] = loss_gs_opa.item()
            loss = loss + args.lambda_gs_opa * loss_gs_opa

        # Normal Consistency in 2dgs
        if args.lambda_normal_consistency > 0:
            lambda_normal = args.lambda_normal_consistency if iteration > 7000 else 0.0
            surf_normal = depth_to_normal(depth, args.vfov, args.hfov)
            render_normal = render_pkg["normal"]
            loss_normal_consistency = (1 - (render_normal * surf_normal).sum(dim=0)[1:-1, 1:-1]).mean()
            log_dict['loss_normal_consistency'] = loss_normal_consistency.item()
            loss = loss + lambda_normal * loss_normal_consistency

        if args.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o * torch.log(o)).mean()
            log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
            loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

        if args.lambda_depth_var > 0:
            depth_var = render_pkg["depth_square"] - depth ** 2
            loss_depth_var = depth_var.clamp_min(1e-6).sqrt().mean()
            log_dict["loss_depth_var"] = loss_depth_var.item()
            lambda_depth_var = args.lambda_depth_var if iteration > 3000 else 0.0
            loss = loss + lambda_depth_var * loss_depth_var

        loss.backward()
        log_dict['loss'] = loss.item()

        iter_end.record()

        with torch.no_grad():
            if conf_trackers:
                scale = scene.resolution_scales[scene.scale_index]
                towards = viewpoint_cam.towards or 'forward'
                if scale in conf_trackers and towards in conf_trackers[scale]:
                    pts_depth_conf = viewpoint_cam.pts_depth.cuda()
                    od_mask_conf = getattr(viewpoint_cam, 'od_loss_mask', None)
                    if od_mask_conf is not None:
                        od_mask_conf = od_mask_conf.cuda()
                        if od_mask_conf.shape[-2:] != pts_depth_conf.shape[-2:]:
                            od_mask_conf = F.interpolate(
                                od_mask_conf.unsqueeze(0),
                                size=pts_depth_conf.shape[-2:],
                                mode='nearest',
                            ).squeeze(0)
                    conf_trackers[scale][towards].update(pts_depth_conf, od_mask_conf)

            for key in (['loss', "loss_l1", "psnr"] if not args.only_velodyne else ['loss']):
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]

            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k: f"{ema_dict_for_log[k]:.{5}f}" for k, v in ema_dict_for_log.items()}
                postfix["scale"] = scene.resolution_scales[scene.scale_index]
                postfix["points_num"] = gaussians.get_xyz.shape[0]
                progress_bar.set_postfix(postfix)

            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            if training_logger is not None:
                training_logger.log_train_step(iteration, log_dict)

            # Save before eval so an OOM in complete_eval/refine does not lose this iter.
            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt/chkpnt" + str(iteration) + ".pth")
                torch.save((lidar_raydrop_prior.capture(), iteration), scene.model_path + "/ckpt/lidar_raydrop_prior_chkpnt" + str(iteration) + ".pth")

            complete_eval(iteration, args.test_iterations, scene, render, (args, background),
                          log_dict, env_map=lidar_raydrop_prior)

            # Densification
            if iteration > args.densify_until_iter * args.time_split_frac:
                gaussians.no_time_split = False

            if iteration < args.densify_until_iter and (args.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < args.densify_until_num_points):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None

                    if size_threshold is not None:
                        size_threshold = size_threshold // scene.resolution_scales[0]

                    gaussians.densify_and_prune(args.densify_grad_threshold, args.densify_grad_abs_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold,
                                                args.densify_grad_t_threshold)

                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            lidar_raydrop_prior.optimizer.step()
            lidar_raydrop_prior.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            if iteration % args.vis_step == 0 or iteration == 1:
                other_img = []

                depth_another = render_pkg['depth_mean'] if args.median_depth else render_pkg['depth_median']
                other_img.append(visualize_depth(depth_another, scale_factor=args.scale_factor))

                if viewpoint_cam.pts_depth is not None:
                    pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth, scale_factor=args.scale_factor)
                    other_img.append(pts_depth_vis)

                feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                v_map = feature[1:4]
                v_norm_map = v_map.norm(dim=0, keepdim=True)
                v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                other_img.append(v_color)

                if args.lambda_raydrop > 0:
                    raydrop_map = render_pkg['raydrop']
                    raydrop_map = visualize_depth(raydrop_map, near=0.01, far=1)
                    other_img.append(raydrop_map)

                    gt_raydrop = get_raydrop_gt(viewpoint_cam)
                    gt_raydrop = visualize_depth(gt_raydrop, near=0.01, far=1)
                    other_img.append(gt_raydrop)

                if viewpoint_cam.pts_intensity is not None:
                    intensity_sh_map = render_pkg['intensity_sh']
                    intensity_sh_map = intensity_sh_map * mask
                    intensity_sh_map = intensity_sh_map.clamp(0.0, 1.0).repeat(3, 1, 1)
                    other_img.append(intensity_sh_map)

                    mask = (viewpoint_cam.pts_depth.cuda() > 0).float()
                    pts_intensity_vis = viewpoint_cam.pts_intensity.cuda().clamp(0.0, 1.0).repeat(3, 1, 1)
                    other_img.append(pts_intensity_vis)

                if args.lambda_normal_consistency > 0:
                    other_img.append(render_normal / 2 + 0.5)
                    other_img.append(surf_normal / 2 + 0.5)

                if args.lambda_edge_guidance > 0:
                    gt_x_grad = visualize_depth(gt_x_grad / gt_x_grad.max(), near=0.01, far=1)
                    other_img.append(gt_x_grad)

                depth_var = render_pkg["depth_square"] - depth ** 2
                depth_var = depth_var / depth_var.max()
                depth_var = visualize_depth(depth_var, near=0.01, far=1)
                other_img.append(depth_var)

                if args.lambda_distortion > 0:
                    distortion = distortion / distortion.max()
                    distortion = visualize_depth(distortion, near=0.01, far=1)
                    other_img.append(distortion)

                grid = make_grid([visualize_depth(depth, scale_factor=args.scale_factor),
                                  ] + other_img, nrow=4)

                save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
                if training_logger is not None and getattr(args, "logger_log_images", True):
                    training_logger.log_image("train/vis", grid, iteration)

            if iteration % args.scale_increase_interval == 0:
                scene.upScale()
                next_w, next_h = scene.getWH()
                lidar_raydrop_prior.upscale(next_h, next_w)

    if conf_trackers:
        conf_out = {}
        for scale, trackers in conf_trackers.items():
            conf_out[f'forward_{scale}'] = trackers['forward'].counts
            conf_out[f'backward_{scale}'] = trackers['backward'].counts
        np.savez(os.path.join(args.model_path, 'static_confidence.npz'), **conf_out)

    return

def complete_eval(iteration, test_iterations, scene: Scene, renderFunc, renderArgs, log_dict, env_map=None):
    if iteration in test_iterations or iteration == 1:
        scale = scene.resolution_scales[scene.scale_index]
        if iteration < args.iterations:
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},)
        else:
            if args.scene_type == "KittiMot":
                num = len(scene.getTrainCameras()) // 2
                eval_train_frame = num // 5
                traincamera = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id)
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                      {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:] + traincamera[num:][-eval_train_frame:]})
            else:
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                      {'name': 'train', 'cameras': scene.getTrainCameras()})

        h, w = args.hw
        h //= scale
        w //= scale

        half_only = int(getattr(args, "cam_num", 2) or 2) == 1
        pm_hfov = tuple(args.hfov) if half_only else None

        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),  # for intensity sh
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov, hfov=pm_hfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov, hfov=pm_hfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov, hfov=pm_hfov),
        ]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for metric in metrics:
                    metric.clear()

                outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
                os.makedirs(outdir, exist_ok=True)

                hole_stats = []
                n_cams = len(config['cameras'])
                n_items = n_cams if half_only else n_cams // 2
                for idx in range(n_items):
                    if half_only:
                        cam_front = config['cameras'][idx]
                        cam_back = None
                    else:
                        cam_front = config['cameras'][idx * 2]
                        cam_back = config['cameras'][idx * 2 + 1]

                    depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano \
                        = render_range_map(args, cam_front, cam_back, scene.gaussians, renderFunc, renderArgs, env_map, [h, w])

                    raydrop_pano_mask = torch.where(raydrop_pano > 0.5, 1, 0)
                    gt_input_pano = stitch_camera_depth(cam_front, cam_back, [h, w], "pts_depth_input")
                    if gt_input_pano is not None:
                        gt_raydrop_pano = torch.where(gt_input_pano > 0, 0, 1)
                    else:
                        gt_raydrop_pano = torch.where(gt_depth_pano > 0, 0, 1)
                    od_valid_pano = None
                    if getattr(args, 'od_mask_gt_loss', False):
                        od_valid_pano = stitch_od_mask_pano(cam_front, cam_back, [h, w]) > 0.5
                    ego_body_pano = stitch_camera_depth(cam_front, cam_back, [h, w], "ego_body_mask")
                    raydrop_valid = od_valid_pano
                    if ego_body_pano is not None:
                        ego_ok = ego_body_pano <= 0.5
                        raydrop_valid = ego_ok if raydrop_valid is None else (raydrop_valid & ego_ok)
                    dead_pano = stitch_camera_mask(cam_front, cam_back, [h, w], args, "blind_dead_mask")
                    dead_ignore_pano = stitch_camera_mask(cam_front, cam_back, [h, w], args, "blind_ignore_mask")
                    if dead_ignore_pano is not None:
                        dead_ok = dead_ignore_pano <= 0.5
                        raydrop_valid = dead_ok if raydrop_valid is None else (raydrop_valid & dead_ok)

                    if iteration == args.iterations:
                        savedir = os.path.join(args.model_path, "ray_drop_datasets")
                        torch.save(torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]]]), os.path.join(savedir, f"render_{config['name']}", f"{cam_front.colmap_id:03d}.pt"))
                        gt_pack = [gt_raydrop_pano, gt_intensity_pano, gt_depth_pano]
                        if od_valid_pano is not None:
                            gt_pack.append(od_valid_pano.float())
                        torch.save(torch.cat(gt_pack), os.path.join(savedir, f"gt", f"{cam_front.colmap_id:03d}.pt"))
                        # GT AABB body RV for re-paste after UNet refine.
                        if getattr(args, "ego_body_infer_replace", True):
                            from utils.ego_body_mask import build_gt_ego_body_rv
                            dilate = int(getattr(args, "ego_body_replace_dilate", 2) or 0)
                            body_d, body_i, body_clear = build_gt_ego_body_rv(
                                cam_front, cam_back, [h, w], args.vfov, args.hfov,
                                dilate=dilate, device="cpu")
                            if body_d is not None:
                                ego_dir = os.path.join(savedir, "ego_body")
                                os.makedirs(ego_dir, exist_ok=True)
                                torch.save(
                                    torch.cat([body_d.cpu(), body_i.cpu(), body_clear.cpu()]),
                                    os.path.join(ego_dir, f"{cam_front.colmap_id:03d}.pt"))
                        # [dead, ignore]: refine ignores both in BCE and forces dead to drop.
                        if dead_pano is not None and dead_ignore_pano is not None:
                            dead_dir = os.path.join(savedir, "blind_dead")
                            os.makedirs(dead_dir, exist_ok=True)
                            torch.save(
                                torch.cat([dead_pano.cpu(), dead_ignore_pano.cpu()]),
                                os.path.join(dead_dir, f"{cam_front.colmap_id:03d}.pt"))

                    depth_pano = depth_pano * (1.0 - raydrop_pano_mask)
                    intensity_sh_pano = intensity_sh_pano * (1.0 - raydrop_pano_mask)
                    if od_valid_pano is not None:
                        depth_eval = depth_pano * od_valid_pano.float()
                        intensity_eval = intensity_sh_pano * od_valid_pano.float()
                    else:
                        depth_eval = depth_pano
                        intensity_eval = intensity_sh_pano
                    hole_stats.append(compute_hole_stats(depth_eval[[0]], gt_depth_pano, od_valid_pano))

                    grid = [visualize_depth(depth_pano[[0]], scale_factor=args.scale_factor),
                            intensity_sh_pano.clamp(0.0, 1.0).repeat(3, 1, 1),
                            visualize_depth(depth_pano[[1]], scale_factor=args.scale_factor),
                            gt_intensity_pano.clamp(0.0, 1.0).repeat(3, 1, 1),
                            visualize_depth(depth_pano[[2]], scale_factor=args.scale_factor),
                            visualize_depth(raydrop_pano_mask, near=0.01, far=1),
                            visualize_depth(gt_depth_pano, scale_factor=args.scale_factor),
                            visualize_depth(gt_raydrop_pano, near=0.01, far=1)]
                    grid = make_grid(grid, nrow=2)
                    save_image(grid, os.path.join(outdir, f"{cam_front.colmap_id:03d}.png"))

                    static_valid = (gt_depth_pano > 0)
                    if od_valid_pano is not None:
                        static_valid = static_valid & od_valid_pano
                    for i, metric in enumerate(metrics):
                        if i == 0:  # hard code
                            metric.update(raydrop_pano, gt_raydrop_pano, valid_mask=raydrop_valid)
                        elif i == 1:
                            metric.update(intensity_eval, gt_intensity_pano, valid_mask=static_valid)
                        elif i == 2:
                            metric.update(depth_eval[[0]], gt_depth_pano, valid_mask=static_valid)
                        else:
                            metric.update(depth_eval[[i - 3]], gt_depth_pano)

                # Ray drop
                RMSE, Acc, F1 = metrics[0].measure()
                # Intensity sh
                rmse_i_sh, medae_i_sh, lpips_loss_i_sh, ssim_i_sh, psnr_i_sh = metrics[1].measure()
                # depth
                rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
                C_D_mix, F_score_mix = metrics[3].measure().astype(float)
                C_D_mean, F_score_mean = metrics[4].measure().astype(float)
                C_D_median, F_score_median = metrics[5].measure().astype(float)

                hole_payload = {}
                if hole_stats:
                    keys = hole_stats[0].keys()
                    hole_payload = {k: float(np.mean([s[k] for s in hole_stats])) for k in keys}

                with open(os.path.join(outdir, "metrics.json"), "w") as f:
                    metrics_payload = {"split": config['name'], "iteration": iteration,
                               "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                               "Point Cloud mix": {"C-D": C_D_mix, "F-score": F_score_mix},
                               "Point Cloud mean": {"C-D": C_D_mean, "F-score": F_score_mean},
                               "Point Cloud median": {"C-D": C_D_median, "F-score": F_score_median},
                               "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                               "Intensity SH": {"RMSE": rmse_i_sh, "MedAE": medae_i_sh, "LPIPS": lpips_loss_i_sh, "SSIM": ssim_i_sh, "PSNR": psnr_i_sh},
                               }
                    if hole_payload:
                        metrics_payload["Holes"] = hole_payload
                    json.dump(metrics_payload, f, indent=1)

                if training_logger is not None:
                    training_logger.log_eval_metrics(iteration, config['name'], {
                        "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                        "Point Cloud mix": {"C-D": C_D_mix, "F-score": F_score_mix},
                        "Point Cloud mean": {"C-D": C_D_mean, "F-score": F_score_mean},
                        "Point Cloud median": {"C-D": C_D_median, "F-score": F_score_median},
                        "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                        "Intensity SH": {"RMSE": rmse_i_sh, "MedAE": medae_i_sh, "LPIPS": lpips_loss_i_sh, "SSIM": ssim_i_sh, "PSNR": psnr_i_sh},
                    })

        torch.cuda.empty_cache()


def refine():
    refine_output_dir = os.path.join(args.model_path, "refine")
    # if os.path.exists(refine_output_dir):
        # shutil.rmtree(refine_output_dir)
    os.makedirs(refine_output_dir, exist_ok=True)
    gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")
    train_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_train")

    unet = UNet(in_channels=3, out_channels=1)
    unet.cuda()
    unet.train()

    raydrop_input_list = []
    raydrop_gt_list = []
    raydrop_valid_list = []

    train_names = [n for n in os.listdir(train_dir) if n.endswith(".pt")]
    if not train_names:
        print(f"Skip refine: no raydrop train tensors in {train_dir}. "
              "Need a last-iteration eval to write ray_drop_datasets.")
        return

    dead_dir = os.path.join(args.model_path, "ray_drop_datasets", "blind_dead")
    use_dead = dead_mask_enabled(args)

    print("Preparing for Raydrop Refinemet ...")
    for data in tqdm(train_names):
        raydrop_input = torch.load(os.path.join(train_dir, data)).unsqueeze(0)
        raydrop_input_list.append(raydrop_input)
        gt_pack = torch.load(os.path.join(gt_dir, data))
        gt_raydrop = gt_pack[[0]].unsqueeze(0)
        raydrop_gt_list.append(gt_raydrop)
        if gt_pack.shape[0] > 3:
            valid = gt_pack[[3]].unsqueeze(0)
        else:
            valid = torch.ones_like(gt_raydrop)
        dead_path = os.path.join(dead_dir, data)
        if use_dead and os.path.isfile(dead_path):
            valid = valid * (1.0 - torch.load(dead_path, map_location=valid.device)[[1]].unsqueeze(0))
        raydrop_valid_list.append(valid)

    torch.cuda.empty_cache()

    # Keep the full stack on CPU. Multi-sensor QT128 can exceed 1700 pans;
    # moving only the batch to GPU avoids OOM during refine().
    raydrop_input = torch.cat(raydrop_input_list, dim=0).float().contiguous()  # [B, 3, H, W]
    raydrop_gt = torch.cat(raydrop_gt_list, dim=0).float().contiguous()  # [B, 1, H, W]
    raydrop_valid = torch.cat(raydrop_valid_list, dim=0).float().contiguous()  # [B, 1, H, W]
    del raydrop_input_list, raydrop_gt_list, raydrop_valid_list
    print(f"Refine stack on CPU: input={tuple(raydrop_input.shape)} "
          f"({raydrop_input.nbytes / 1e9:.2f} GB) + gt/valid")

    loss_total = []

    refine_bs = 32  # set smaller batch size (e.g. 32) if OOM and adjust epochs accordingly
    refine_all_steps = 5000

    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=refine_all_steps)
    bce_fn = torch.nn.BCELoss()

    print("Start UNet Optimization ...")
    for i in range(refine_all_steps):
        optimizer.zero_grad()

        if refine_bs is not None:
            idx = np.random.choice(raydrop_input.shape[0], refine_bs, replace=False)
            input = raydrop_input[idx, ...].cuda(non_blocking=True)
            gt = raydrop_gt[idx, ...].cuda(non_blocking=True)
            valid = raydrop_valid[idx, ...].cuda(non_blocking=True)
        else:
            input = raydrop_input.cuda()
            gt = raydrop_gt.cuda()
            valid = raydrop_valid.cuda()

        # random mask
        mask = torch.ones_like(input).to(input.device)
        box_num_max = 32
        box_size_y_max = int(0.1 * input.shape[2])
        box_size_x_max = int(0.1 * input.shape[3])
        for j in range(np.random.randint(box_num_max)):
            box_size_y = np.random.randint(1, box_size_y_max)
            box_size_x = np.random.randint(1, box_size_x_max)
            yi = np.random.randint(input.shape[2] - box_size_y)
            xi = np.random.randint(input.shape[3] - box_size_x)
            mask[:, :, yi:yi + box_size_y, xi:xi + box_size_x] = 0.

        raydrop_refine = unet(input * mask)
        if valid.min() < 1:
            bce_map = torch.nn.functional.binary_cross_entropy(raydrop_refine, gt, reduction='none')
            bce_loss = (bce_map * valid).sum() / valid.sum().clamp_min(1)
        else:
            bce_loss = bce_fn(raydrop_refine, gt)
        loss = bce_loss

        loss.backward()

        loss_total.append(loss.item())

        if i % 50 == 0:
            input_mask = torch.where(input > 0.5, 1, 0)
            raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
            idx = np.random.choice(range(raydrop_mask.shape[0]))
            grid = [visualize_depth(input_mask[idx], near=0.01, far=1),
                    visualize_depth(raydrop_mask[idx], near=0.01, far=1),
                    visualize_depth(gt[idx], near=0.01, far=1)]
            grid = make_grid(grid, nrow=1)
            save_image(grid, os.path.join(refine_output_dir, f"{i:04d}.png"))
            log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{log_time}] iter:{i}, lr:{optimizer.param_groups[0]['lr']:.6f}, raydrop loss:{loss.item()}")
            if training_logger is not None:
                training_logger.log_scalars(
                    {"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']},
                    i,
                    prefix="refine",
                )

        optimizer.step()
        scheduler.step()

    file_path = f"{args.model_path}/ckpt/refine.pth"
    torch.save(unet.state_dict(), file_path)

    torch.cuda.empty_cache()


def refine_test():
    refine_test_bs = 32  # reduce if OOM

    torch.cuda.empty_cache()

    file_path = f"{args.model_path}/ckpt/refine.pth"
    if not os.path.exists(file_path):
        print(f"Skip refine_test: {file_path} not found.")
        return
    unet = UNet(in_channels=3, out_channels=1)
    unet.load_state_dict(torch.load(file_path, map_location="cpu"))
    unet.cuda()
    unet.eval()

    # cam_num=1 saves HxW half pans with args.hfov; cam_num=2 saves stitched Hx(2W) ±180°.
    half_only = int(getattr(args, "cam_num", 2) or 2) == 1
    hfov = tuple(args.hfov) if half_only else (-180, 180)
    pm_hfov = tuple(args.hfov) if half_only else None

    for mode in ["train", "test"]:
        outdir = os.path.join(args.model_path, "eval", f"{mode}_refine_render")
        os.makedirs(outdir, exist_ok=True)

        test_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_{mode}")
        gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")
        name_list = sorted(os.listdir(test_dir))

        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov, hfov=pm_hfov),
        ]

        print(f"Refine test {mode}: {len(name_list)} frames, batch size {refine_test_bs}")
        hole_stats = []
        with torch.inference_mode():
            for batch_start in tqdm(range(0, len(name_list), refine_test_bs)):
                batch_names = name_list[batch_start:batch_start + refine_test_bs]
                test_input = torch.cat(
                    [torch.load(os.path.join(test_dir, name), map_location="cpu").unsqueeze(0).float()
                     for name in batch_names],
                    dim=0,
                ).cuda(non_blocking=True)
                gt = torch.cat(
                    [torch.load(os.path.join(gt_dir, name), map_location="cpu").unsqueeze(0).float()
                     for name in batch_names],
                    dim=0,
                ).cuda(non_blocking=True)

                raydrop_refine = unet(test_input)
                raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)

                ego_body_dir = os.path.join(args.model_path, "ray_drop_datasets", "ego_body")
                dead_dir = os.path.join(args.model_path, "ray_drop_datasets", "blind_dead")
                for idx, name in enumerate(batch_names):
                    raydrop_pano = raydrop_refine[idx, [0]]
                    dead_ok = None
                    dead_path = os.path.join(dead_dir, name)
                    if dead_mask_enabled(args) and os.path.isfile(dead_path):
                        dead_pack = torch.load(dead_path, map_location=raydrop_pano.device)
                        _, _, raydrop_pano = force_dead_drop(None, None, raydrop_pano, dead_pack[[0]])
                        dead_ok = dead_pack[[1]] <= 0.5
                    raydrop_pano_mask = torch.where(raydrop_pano > 0.5, 1, 0)
                    intensity_pano = test_input[idx, [1]] * (1 - raydrop_pano_mask)
                    depth_pano = test_input[idx, [2]] * (1 - raydrop_pano_mask)

                    # Re-paste this frame's GT AABB body RV after UNet raydrop.
                    if getattr(args, "ego_body_infer_replace", True):
                        ego_path = os.path.join(ego_body_dir, name)
                        if os.path.isfile(ego_path):
                            from utils.ego_body_mask import paste_gt_ego_body_rv
                            ego_pack = torch.load(ego_path, map_location=depth_pano.device)
                            if ego_pack.ndim == 3 and ego_pack.shape[0] >= 3:
                                depth_pano, intensity_pano, raydrop_pano = paste_gt_ego_body_rv(
                                    depth_pano, intensity_pano, raydrop_pano,
                                    ego_pack[[0]], ego_pack[[1]], ego_pack[[2]])

                    gt_raydrop_pano = gt[idx, [0]]
                    gt_intensity_pano = gt[idx, [1]]
                    gt_depth_pano = gt[idx, [2]]
                    od_valid = gt[idx, [3]] > 0.5 if gt.shape[1] > 3 else None
                    if od_valid is not None:
                        depth_eval = depth_pano * od_valid.float()
                        intensity_eval = intensity_pano * od_valid.float()
                    else:
                        depth_eval = depth_pano
                        intensity_eval = intensity_pano
                    static_valid = gt_depth_pano > 0
                    if od_valid is not None:
                        static_valid = static_valid & od_valid
                    hole_stats.append(compute_hole_stats(depth_eval, gt_depth_pano, od_valid))

                    grid = [visualize_depth(gt_depth_pano, scale_factor=args.scale_factor),
                            visualize_depth(depth_eval, scale_factor=args.scale_factor),
                            gt_intensity_pano.clamp(0, 1).repeat(3, 1, 1),
                            intensity_eval.clamp(0, 1).repeat(3, 1, 1), ]
                    grid = make_grid(grid, nrow=1, padding=0)
                    save_image(grid, os.path.join(outdir, name.replace(".pt", ".png")))
                    mask_ = depth_eval > 0
                    intensity_ = intensity_eval[mask_].reshape(-1, 1)
                    save_ply(pano_to_lidar(depth_eval, args.vfov, hfov),
                             os.path.join(outdir, name.replace(".pt", ".ply")), intensity_)
                    mask_ = gt_depth_pano > 0
                    intensity_ = gt_intensity_pano[mask_].reshape(-1, 1)
                    save_ply(pano_to_lidar(gt_depth_pano, args.vfov, hfov),
                             os.path.join(outdir, name.replace(".pt", "_gt.ply")), intensity_)

                    for i, metric in enumerate(metrics):
                        if i == 0:  # hard code
                            raydrop_valid = od_valid
                            if dead_ok is not None:
                                raydrop_valid = dead_ok if raydrop_valid is None else (raydrop_valid & dead_ok)
                            metric.update(raydrop_pano, gt_raydrop_pano, valid_mask=raydrop_valid)
                        elif i == 1:
                            metric.update(intensity_eval, gt_intensity_pano, valid_mask=static_valid)
                        elif i == 2:
                            metric.update(depth_eval, gt_depth_pano, valid_mask=static_valid)
                        else:
                            metric.update(depth_eval, gt_depth_pano)

                del test_input, gt, raydrop_refine, raydrop_mask
                torch.cuda.empty_cache()

        # Ray drop
        RMSE, Acc, F1 = metrics[0].measure()
        # Intensity
        rmse_i, medae_i, lpips_loss_i, ssim_i, psnr_i = metrics[1].measure()
        # depth
        rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
        C_D, F_score = metrics[3].measure().astype(float)

        hole_payload = {}
        if hole_stats:
            keys = hole_stats[0].keys()
            hole_payload = {k: float(np.mean([s[k] for s in hole_stats])) for k in keys}

        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            refine_metrics = {"split": f"{mode}", "iteration": "refine",
                       "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                       "Point Cloud": {"C-D": C_D, "F-score": F_score},
                       "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                       "Intensity": {"RMSE": rmse_i, "MedAE": medae_i, "LPIPS": lpips_loss_i, "SSIM": ssim_i, "PSNR": psnr_i},
                       }
            if hole_payload:
                refine_metrics["Holes"] = hole_payload
            json.dump(refine_metrics, f, indent=1)

        if training_logger is not None:
            training_logger.log_eval_metrics(args.iterations, f"{mode}_refine", {
                "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                "Point Cloud": {"C-D": C_D, "F-score": F_score},
                "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                "Intensity": {"RMSE": rmse_i, "MedAE": medae_i, "LPIPS": lpips_loss_i, "SSIM": ssim_i, "PSNR": psnr_i},
            })

    del unet
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--debug_cuda", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--refine_only", action="store_true",
                        help="Skip GS training; run UNet refine (+ refine_test) on existing ray_drop_datasets")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument("--show_log", action="store_true",
                        help="Print training log to terminal (default: write to model_path/log.txt)")
    args_read, cli_unknown = parser.parse_known_args()

    base_conf = OmegaConf.load(args_read.base_config)
    second_conf = OmegaConf.load(args_read.config)
    cli_dotlist = [u for u in cli_unknown if "=" in u]
    cli_conf = OmegaConf.from_dotlist(cli_dotlist)
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    OmegaConf.update(args, "start_checkpoint", args_read.start_checkpoint)
    OmegaConf.update(args, "debug_cuda", args_read.debug_cuda)
    OmegaConf.update(args, "test_only", args_read.test_only)
    OmegaConf.update(args, "median_depth", args_read.median_depth)

    # if os.path.exists(args.model_path) and not args.test_only and args.start_checkpoint is None:
        # shutil.rmtree(args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    if not args.dynamic:
        args.t_grad = False

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if args.test_only:
        args.shuffle = False
        for iteration in args.checkpoint_iterations:
            path = os.path.join(args.model_path, "ckpt", f"chkpnt{iteration}.pth")
            if os.path.exists(path):
                args.start_checkpoint = path
                resolution_idx = len(args.resolution_scales) - 1
                for i in range(iteration // args.scale_increase_interval):
                    resolution_idx = max(0, resolution_idx - 1)
        args.resolution_scales = [args.resolution_scales[resolution_idx]]
        with open(os.path.join(args.model_path, "scale_factor.txt"), 'r') as file:
            data = file.read()
            args.scale_factor = float(data)

    if args.debug_cuda:
        args.resolution_scales = [args.resolution_scales[-1]]

    if args.exhaust_test:
        args.test_iterations += [i for i in range(0, args.iterations, args.test_interval)]

    print(args)

    print("Optimizing " + args.model_path)
    if not args_read.refine_only:
        with open(os.path.join(args.model_path, 'setting.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in args.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

        os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'gt'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_train'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_test'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'ckpt'), exist_ok=True)

    if not args.test_only and not args.debug_cuda and not args_read.show_log:
        log_path = os.path.join(args.model_path, 'log.txt')
        print(f"[GS-LiDAR] Terminal output redirected to {log_path}")
        print("[GS-LiDAR] Use --show_log to print here, or: tail -f", log_path)
        if not args_read.refine_only:
            print("[GS-LiDAR] Loading dataset (GACRT ~206 frames) may take several minutes before the training bar appears.")
        f = open(log_path, 'a' if args_read.refine_only else 'w', buffering=1)
        sys.stdout = f
        sys.stderr = f
    seed_everything(args.seed)

    global training_logger
    if not args.test_only and not args_read.refine_only and args.logger_backend != "none":
        training_logger = TrainingLogger(args)
    else:
        training_logger = None

    if args_read.refine_only:
        print("\n--refine_only: skip GS training, use existing ray_drop_datasets.")
    else:
        training(args)
        torch.cuda.empty_cache()
        print("\nTraining complete.")

    if not args.test_only or args_read.refine_only:
        refine()
        torch.cuda.empty_cache()

    refine_ckpt = os.path.join(args.model_path, "ckpt", "refine.pth")
    if os.path.exists(refine_ckpt):
        refine_test()
        print("\nRefine complete.")
    else:
        print(f"\nSkip refine_test: {refine_ckpt} not found. "
              "Finish the last training eval (writes ray_drop_datasets) then run refine.")
    if training_logger is not None:
        training_logger.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
