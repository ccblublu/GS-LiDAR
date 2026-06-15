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
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from scene.unet import UNet

EPS = 1e-5


def training(args):
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)

    gaussians = GaussianModel(args)

    scene = Scene(args, gaussians, shuffle=args.shuffle)
    with open(os.path.join(args.model_path, 'scale_factor.txt'), 'w') as f:
        f.writelines(str(args.scale_factor))

    gaussians.training_setup(args)

    start_w, start_h = scene.getWH()
    lidar_raydrop_prior = RayDropPrior(h=start_h, w=start_w).cuda()
    lidar_raydrop_prior.training_setup(args)

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)

        lidar_raydrop_prior_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                                      os.path.basename(args.start_checkpoint).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
        (lidar_raydrop_prior_params, _) = torch.load(lidar_raydrop_prior_checkpoint)
        lidar_raydrop_prior.restore(lidar_raydrop_prior_params)

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

            mask = pts_depth > 0
            loss_lidar = F.l1_loss(pts_depth[mask], depth[mask])
            if args.lidar_decay > 0:
                iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
            else:
                iter_decay = 1
            log_dict['loss_lidar'] = loss_lidar.item()
            loss += iter_decay * args.lambda_lidar * loss_lidar

        if args.lambda_lidar_median > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = pts_depth > 0
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
            mask = pts_depth > 0
            pts_intensity = viewpoint_cam.pts_intensity.cuda()
            loss_intensity_sh = torch.nn.functional.l1_loss(pts_intensity[mask], intensity_sh_map[mask])
            log_dict['loss_intensity_sh'] = loss_intensity_sh.item()
            loss += args.lambda_intensity_sh * loss_intensity_sh

        if args.lambda_raydrop > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            gt_raydrop = 1.0 - (pts_depth > 0).float()
            loss_raydrop = torch.nn.functional.binary_cross_entropy(raydrop_map, gt_raydrop)
            log_dict['loss_raydrop'] = loss_raydrop.item()
            loss += args.lambda_raydrop * loss_raydrop

        # chamfer loss
        if args.lambda_chamfer > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = (pts_depth > 0).float()

            cham_fn = chamfer_3DDist()
            pred_lidar = pano_to_lidar(depth * mask, args.vfov, args.hfov) / args.scale_factor
            gt_lidar = pano_to_lidar(pts_depth, args.vfov, args.hfov) / args.scale_factor
            dist1, dist2, _, _ = cham_fn(pred_lidar[None], gt_lidar[None])

            loss_chamfer = dist1.mean() + dist2.mean()
            log_dict['loss_chamfer'] = loss_chamfer.item()
            loss += args.lambda_chamfer * loss_chamfer

        if args.lambda_smooth > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            gt_grad_x = pts_depth[:, :, :-1] - pts_depth[:, :, 1:]
            gt_grad_y = pts_depth[:, :-1, :] - pts_depth[:, 1:, :]
            mask_x = (torch.where(pts_depth[:, :, :-1] > 0, 1, 0) *
                      torch.where(pts_depth[:, :, 1:] > 0, 1, 0))
            mask_y = (torch.where(pts_depth[:, :-1, :] > 0, 1, 0) *
                      torch.where(pts_depth[:, 1:, :] > 0, 1, 0))

            grad_clip = 0.01 * args.scale_factor
            grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip, 1, 0) * mask_x
            grad_mask_x = grad_mask_x.bool()
            grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip, 1, 0) * mask_y
            grad_mask_y = grad_mask_y.bool()

            pred_grad_x = depth[:, :, :-1] - depth[:, :, 1:]
            pred_grad_y = depth[:, :-1, :] - depth[:, 1:, :]
            loss_smooth = (F.l1_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x])
                           + F.l1_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y]))
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
            for key in (['loss', "loss_l1", "psnr"] if not args.only_velodyne else ['loss']):
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]

            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k: f"{ema_dict_for_log[k]:.{5}f}" for k, v in ema_dict_for_log.items()}
                postfix["scale"] = scene.resolution_scales[scene.scale_index]
                postfix["points_num"] = gaussians.get_xyz.shape[0]
                progress_bar.set_postfix(postfix)

            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            # Log and save
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

                    gt_raydrop = 1.0 - (viewpoint_cam.pts_depth.cuda() > 0).float()
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

            if iteration % args.scale_increase_interval == 0:
                scene.upScale()
                next_w, next_h = scene.getWH()
                lidar_raydrop_prior.upscale(next_h, next_w)

            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt/chkpnt" + str(iteration) + ".pth")
                torch.save((lidar_raydrop_prior.capture(), iteration), scene.model_path + "/ckpt/lidar_raydrop_prior_chkpnt" + str(iteration) + ".pth")

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

        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),  # for intensity sh
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov)
        ]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for metric in metrics:
                    metric.clear()

                outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
                os.makedirs(outdir, exist_ok=True)

                for idx in range(len(config['cameras']) // 2):
                    cam_front = config['cameras'][idx * 2]
                    cam_back = config['cameras'][idx * 2 + 1]

                    depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano \
                        = render_range_map(args, cam_front, cam_back, scene.gaussians, renderFunc, renderArgs, env_map, [h, w])

                    raydrop_pano_mask = torch.where(raydrop_pano > 0.5, 1, 0)
                    gt_raydrop_pano = torch.where(gt_depth_pano > 0, 0, 1)

                    if iteration == args.iterations:
                        savedir = os.path.join(args.model_path, "ray_drop_datasets")
                        torch.save(torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]]]), os.path.join(savedir, f"render_{config['name']}", f"{cam_front.colmap_id:03d}.pt"))
                        torch.save(torch.cat([gt_raydrop_pano, gt_intensity_pano, gt_depth_pano]), os.path.join(savedir, f"gt", f"{cam_front.colmap_id:03d}.pt"))

                    depth_pano = depth_pano * (1.0 - raydrop_pano_mask)
                    intensity_sh_pano = intensity_sh_pano * (1.0 - raydrop_pano_mask)

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

                    for i, metric in enumerate(metrics):
                        if i == 0:  # hard code
                            metric.update(raydrop_pano, gt_raydrop_pano)
                        elif i == 1:
                            metric.update(intensity_sh_pano, gt_intensity_pano)
                        elif i == 2:
                            metric.update(depth_pano[[0]], gt_depth_pano)
                        else:
                            metric.update(depth_pano[[i - 3]], gt_depth_pano)

                # Ray drop
                RMSE, Acc, F1 = metrics[0].measure()
                # Intensity sh
                rmse_i_sh, medae_i_sh, lpips_loss_i_sh, ssim_i_sh, psnr_i_sh = metrics[1].measure()
                # depth
                rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
                C_D_mix, F_score_mix = metrics[3].measure().astype(float)
                C_D_mean, F_score_mean = metrics[4].measure().astype(float)
                C_D_median, F_score_median = metrics[5].measure().astype(float)

                with open(os.path.join(outdir, "metrics.json"), "w") as f:
                    json.dump({"split": config['name'], "iteration": iteration,
                               "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                               "Point Cloud mix": {"C-D": C_D_mix, "F-score": F_score_mix},
                               "Point Cloud mean": {"C-D": C_D_mean, "F-score": F_score_mean},
                               "Point Cloud median": {"C-D": C_D_median, "F-score": F_score_median},
                               "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                               "Intensity SH": {"RMSE": rmse_i_sh, "MedAE": medae_i_sh, "LPIPS": lpips_loss_i_sh, "SSIM": ssim_i_sh, "PSNR": psnr_i_sh},
                               }, f, indent=1)

        torch.cuda.empty_cache()


def refine():
    refine_output_dir = os.path.join(args.model_path, "refine")
    if os.path.exists(refine_output_dir):
        shutil.rmtree(refine_output_dir)
    os.makedirs(refine_output_dir)
    gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")
    train_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_train")

    unet = UNet(in_channels=3, out_channels=1)
    unet.cuda()
    unet.train()

    raydrop_input_list = []
    raydrop_gt_list = []

    print("Preparing for Raydrop Refinemet ...")
    for data in tqdm(os.listdir(train_dir)):
        raydrop_input = torch.load(os.path.join(train_dir, data)).unsqueeze(0)
        raydrop_input_list.append(raydrop_input)
        gt_raydrop = torch.load(os.path.join(gt_dir, data))[[0]].unsqueeze(0)
        raydrop_gt_list.append(gt_raydrop)

    torch.cuda.empty_cache()

    raydrop_input = torch.cat(raydrop_input_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]
    raydrop_gt = torch.cat(raydrop_gt_list, dim=0).cuda().float().contiguous()  # [B, 1, H, W]

    loss_total = []

    refine_bs = None  # set smaller batch size (e.g. 32) if OOM and adjust epochs accordingly
    refine_epoch = 1000

    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=refine_epoch)
    bce_fn = torch.nn.BCELoss()

    print("Start UNet Optimization ...")
    for i in range(refine_epoch):
        optimizer.zero_grad()

        if refine_bs is not None:
            idx = np.random.choice(raydrop_input.shape[0], refine_bs, replace=False)
            input = raydrop_input[idx, ...]
            gt = raydrop_gt[idx, ...]
        else:
            input = raydrop_input
            gt = raydrop_gt

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

        optimizer.step()
        scheduler.step()

    file_path = f"{args.model_path}/ckpt/refine.pth"
    torch.save(unet.state_dict(), file_path)

    torch.cuda.empty_cache()


def refine_test():
    file_path = f"{args.model_path}/ckpt/refine.pth"
    unet = UNet(in_channels=3, out_channels=1)
    unet.load_state_dict(torch.load(file_path))
    unet.cuda()
    unet.eval()

    for mode in ["train", "test"]:
        outdir = os.path.join(args.model_path, "eval", f"{mode}_refine_render")
        os.makedirs(outdir, exist_ok=True)

        test_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_{mode}")
        gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")

        test_input_list = []
        gt_list = []
        name_list = []
        print(f"Preparing for Refinemet {mode} ...")
        for data in tqdm(os.listdir(test_dir)):
            raydrop_input = torch.load(os.path.join(test_dir, data)).unsqueeze(0)
            test_input_list.append(raydrop_input)
            gt_raydrop = torch.load(os.path.join(gt_dir, data)).unsqueeze(0)
            gt_list.append(gt_raydrop)
            name_list.append(data)

        test_input = torch.cat(test_input_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]
        gt = torch.cat(gt_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]

        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov)
        ]

        with torch.no_grad():
            raydrop_refine = unet(test_input)
            raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
            for idx in tqdm(range(gt.shape[0])):
                raydrop_pano = raydrop_refine[idx, [0]]
                raydrop_pano_mask = raydrop_mask[idx, [0]]
                intensity_pano = test_input[idx, [1]] * (1 - raydrop_pano_mask)
                depth_pano = test_input[idx, [2]] * (1 - raydrop_pano_mask)

                gt_raydrop_pano = gt[idx, [0]]
                gt_intensity_pano = gt[idx, [1]]
                gt_depth_pano = gt[idx, [2]]

                grid = [visualize_depth(gt_depth_pano, scale_factor=args.scale_factor),
                        visualize_depth(depth_pano, scale_factor=args.scale_factor),
                        gt_intensity_pano.clamp(0, 1).repeat(3, 1, 1),
                        intensity_pano.clamp(0, 1).repeat(3, 1, 1), ]
                grid = make_grid(grid, nrow=1, padding=0)
                save_image(grid, os.path.join(outdir, name_list[idx].replace(".pt", ".png")))
                save_ply(pano_to_lidar(depth_pano, args.vfov, (-180, 180)),
                         os.path.join(outdir, name_list[idx].replace(".pt", ".ply")))

                for i, metric in enumerate(metrics):
                    if i == 0:  # hard code
                        metric.update(raydrop_pano, gt_raydrop_pano)
                    elif i == 1:
                        metric.update(intensity_pano, gt_intensity_pano)
                    else:
                        metric.update(depth_pano, gt_depth_pano)

            # Ray drop
            RMSE, Acc, F1 = metrics[0].measure()
            # Intensity
            rmse_i, medae_i, lpips_loss_i, ssim_i, psnr_i = metrics[1].measure()
            # depth
            rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
            C_D, F_score = metrics[3].measure().astype(float)

            with open(os.path.join(outdir, "metrics.json"), "w") as f:
                json.dump({"split": f"{mode}", "iteration": "refine",
                           "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                           "Point Cloud": {"C-D": C_D, "F-score": F_score},
                           "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                           "Intensity": {"RMSE": rmse_i, "MedAE": medae_i, "LPIPS": lpips_loss_i, "SSIM": ssim_i, "PSNR": psnr_i},
                           }, f, indent=1)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--debug_cuda", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument("--show_log", action="store_true")
    args_read, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args_read.base_config)
    second_conf = OmegaConf.load(args_read.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    OmegaConf.update(args, "start_checkpoint", args_read.start_checkpoint)
    OmegaConf.update(args, "debug_cuda", args_read.debug_cuda)
    OmegaConf.update(args, "test_only", args_read.test_only)
    OmegaConf.update(args, "median_depth", args_read.median_depth)

    if os.path.exists(args.model_path) and not args.test_only and args.start_checkpoint is None:
        shutil.rmtree(args.model_path)
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
    with open(os.path.join(args.model_path, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    if os.path.exists(os.path.join(args.model_path, 'ray_drop_datasets')) and not args.test_only:
        shutil.rmtree(os.path.join(args.model_path, 'ray_drop_datasets'))
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_train'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_test'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ckpt'), exist_ok=True)

    if not args.test_only and not args.debug_cuda and not args_read.show_log:
        f = open(os.path.join(args.model_path, 'log.txt'), 'w')
        sys.stdout = f
        sys.stderr = f
    seed_everything(args.seed)

    if not args.test_only:
        training(args)
        torch.cuda.empty_cache()
    # Training done
    print("\nTraining complete.")

    if not args.test_only:
        refine()
        torch.cuda.empty_cache()

    refine_test()
    print("\nRefine complete.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
