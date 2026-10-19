#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import torch
import math
from .diff_gaussian_rasterization_2d import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh


def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, env_map=None,
           time_shift=None, other=[], mask=None, is_training=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if pipe.neg_fov:
        # we find that set fov as -1 slightly improves the results
        tanfovx = math.tan(-0.5)
        tanfovy = math.tan(-0.5)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        vfov=viewpoint_camera.vfov,
        hfov=viewpoint_camera.hfov,
        scale_factor=pipe.scale_factor,
        near_dist=float(getattr(pipe, "raster_near", 2.0)),
    )

    assert raster_settings.bg.shape[0] == 4

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None

    if time_shift is not None:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - time_shift)
        means3D = means3D + pc.get_inst_velocity * time_shift
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - time_shift)
    else:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)

    # 静态不乘 marginal_t 了
    if pipe.dynamic:
        opacity = opacity * marginal_t

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    feature_list = other

    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        S_other = 0

    # Prefilter
    mask = (opacity[:, 0] > 1 / 255) if mask is None else mask & (opacity[:, 0] > 1 / 255)
    if pipe.dynamic:
        # Static Gaussians (is_dynamic=False) are exempt from temporal mask
        if pc._is_dynamic.shape[0] > 0 and pc._is_dynamic.shape[0] == opacity.shape[0]:
            temporal_mask = (marginal_t[:, 0] > 0.05) | (~pc.get_is_dynamic[:, 0])
        else:
            temporal_mask = marginal_t[:, 0] > 0.05
        mask = mask & temporal_mask

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, rendered_depth, rendered_opacity, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=mask)

    _, rendered_intensity_sh, rendered_raydrop = rendered_image.split([2, 1, 1], dim=0)
    rendered_other, rendered_normal = rendered_feature.split([S_other, 3], dim=0)
    rendered_normal = rendered_normal / (rendered_normal.norm(dim=0, keepdim=True) + 1e-8)

    # Ring RV applies the prior per ring slot (ring_raydrop_maps), not per grid pixel.
    if env_map is not None and not getattr(pipe, "ring_rv", False):
        lidar_raydrop_prior_from_envmap = env_map(
            viewpoint_camera.towards, getattr(viewpoint_camera, "sensor_id", 0))
        rendered_raydrop = lidar_raydrop_prior_from_envmap + (1 - lidar_raydrop_prior_from_envmap) * rendered_raydrop

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "contrib": contrib,
        "depth": rendered_depth[[1]] if pipe.median_depth else rendered_depth[[0]],
        "depth_mean": rendered_depth[[0]],
        "depth_median": rendered_depth[[1]],
        "distortion": rendered_depth[[2]],
        "depth_square": rendered_depth[[3]],
        "alpha": rendered_opacity,
        "feature": rendered_other,
        "normal": rendered_normal,
        "intensity_sh": rendered_intensity_sh,
        "raydrop": rendered_raydrop.clamp(0, 1)
    }


def _render_half_pkg(viewpoint, gaussians, renderFunc, renderArgs, env_map, args, EPS=1e-5):
    """Render one half-pano; returns depth(3,H,W), intensity, raydrop, gt_depth, gt_intensity."""
    depth_gt = viewpoint.pts_depth.cuda() if viewpoint.pts_depth is not None else viewpoint.pts_depth
    intensity_gt = viewpoint.pts_intensity.cuda() if viewpoint.pts_intensity is not None else viewpoint.pts_intensity
    render_pkg = renderFunc(viewpoint, gaussians, *renderArgs, env_map=env_map)

    depth = render_pkg['depth']
    alpha = render_pkg['alpha']
    raydrop_render = render_pkg['raydrop']

    depth_var = render_pkg['depth_square'] - depth ** 2
    depth_median = render_pkg["depth_median"]
    var_quantile = depth_var.median() * 10

    depth_mix = torch.zeros_like(depth)
    depth_mix[depth_var > var_quantile] = depth_median[depth_var > var_quantile]
    depth_mix[depth_var <= var_quantile] = depth[depth_var <= var_quantile]

    depth = torch.cat([depth_mix, depth, depth_median])

    if args.sky_depth:
        sky_depth = 900
        depth = depth / alpha.clamp_min(EPS)
        if args.depth_blend_mode == 0:  # harmonic mean
            depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
        elif args.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth

    intensity_sh = render_pkg['intensity_sh']
    return depth, intensity_sh, raydrop_render, depth_gt, intensity_gt


def _force_blind_dead(args, depth, intensity, raydrop, cam_front, cam_back, hw):
    """Static blind-lidar dead zone always drops; applied before the ego-body paste."""
    from utils.blind_dead_mask import stitch_camera_mask, force_dead_drop
    dead = stitch_camera_mask(cam_front, cam_back, hw, args, "blind_dead_mask", device=raydrop.device)
    return force_dead_drop(depth, intensity, raydrop, dead)


def render_range_map(args, cam_front: Camera, cam_back: Camera, gaussians, renderFunc, renderArgs, env_map, hw):
    """Render a range image for evaluation.

    ``cam_back is None`` (cam_num=1): single 180° half at HxW with args.hfov.
    Otherwise: classic forward+back stitch to Hx(2W) covering ±180°.
    """
    EPS = 1e-5
    h, w = hw

    if cam_back is None:
        assert cam_front.towards in (None, "forward")
        depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano = \
            _render_half_pkg(cam_front, gaussians, renderFunc, renderArgs, env_map, args, EPS)
        depth_pano, intensity_sh_pano, raydrop_pano = _force_blind_dead(
            args, depth_pano, intensity_sh_pano, raydrop_pano, cam_front, None, [h, w])
        if getattr(args, "ego_body_infer_replace", True):
            from utils.ego_body_mask import composite_ego_body_pano
            dilate = int(getattr(args, "ego_body_replace_dilate", 2) or 0)
            depth_pano, intensity_sh_pano, raydrop_pano = composite_ego_body_pano(
                depth_pano, intensity_sh_pano, raydrop_pano,
                cam_front, None, [h, w], args.vfov, args.hfov, dilate=dilate)
        return depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano

    assert cam_front.towards == "forward" and cam_back.towards == "backward"
    n_sensors = int(getattr(args, "n_lidar_sensors", 1) or 1)
    assert cam_back.colmap_id - cam_front.colmap_id == args.frames * n_sensors
    assert getattr(cam_front, "sensor_id", 0) == getattr(cam_back, "sensor_id", 0)

    breaks = (0, w // 2, 3 * w // 2, w * 2)

    depth_pano = torch.zeros([3, h, w * 2]).cuda()
    intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
    raydrop_pano = torch.zeros([1, h, w * 2]).cuda()
    gt_depth_pano = torch.zeros([1, h, w * 2]).cuda()
    gt_intensity_pano = torch.zeros([1, h, w * 2]).cuda()

    for idx, viewpoint in enumerate([cam_front, cam_back]):
        depth, intensity_sh, raydrop_render, depth_gt, intensity_gt = _render_half_pkg(
            viewpoint, gaussians, renderFunc, renderArgs, env_map, args, EPS)

        if idx % 2 == 0:  # 前180度
            depth_pano[:, :, breaks[1]:breaks[2]] = depth
            gt_depth_pano[:, :, breaks[1]:breaks[2]] = depth_gt

            intensity_sh_pano[:, :, breaks[1]:breaks[2]] = intensity_sh
            gt_intensity_pano[:, :, breaks[1]:breaks[2]] = intensity_gt

            raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop_render

            continue
        else:
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

    depth_pano, intensity_sh_pano, raydrop_pano = _force_blind_dead(
        args, depth_pano, intensity_sh_pano, raydrop_pano, cam_front, cam_back, [h, w])
    if getattr(args, "ego_body_infer_replace", True):
        from utils.ego_body_mask import composite_ego_body_pano
        dilate = int(getattr(args, "ego_body_replace_dilate", 2) or 0)
        depth_pano, intensity_sh_pano, raydrop_pano = composite_ego_body_pano(
            depth_pano, intensity_sh_pano, raydrop_pano,
            cam_front, cam_back, [h, w], args.vfov, args.hfov, dilate=dilate)

    return depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano


def ring_raydrop_maps(args, cam, depth, intensity, raydrop, ref_depth, env_map):
    """Sample render-grid maps at ring slots; raydrop gets the ring-space prior.

    Returns depth (C,R,W), intensity (1,R,W), raydrop (1,R,W), RingGeometry.
    """
    from utils.ring_rv import get_geometry, sample_to_ring
    geom = get_geometry(args, cam.sensor_id, int(cam.image_height), int(cam.image_width))
    n_depth = depth.shape[0]
    ring = sample_to_ring(torch.cat([depth, intensity, raydrop]), ref_depth, geom,
                          float(getattr(args, "ring_interp_rel_thresh", 0.1)))
    ring_raydrop = ring[n_depth + 1:]
    if env_map is not None:
        prior = env_map(cam.towards, getattr(cam, "sensor_id", 0))
        ring_raydrop = prior + (1 - prior) * ring_raydrop
    return ring[:n_depth], ring[n_depth:n_depth + 1], ring_raydrop.clamp(0, 1), geom


def render_ring_range_map(args, cam: Camera, gaussians, renderFunc, renderArgs, env_map):
    """Ring-RV inference for one blind-lidar camera (cam_num=1).

    Returns depth (3,R,W) [mix, mean, median], intensity, raydrop, gt_depth,
    gt_intensity (all ring space) and the RingGeometry. Unfired placeholder
    slots and the static dead zone always drop; GT body returns are pasted.
    """
    depth, intensity_sh, raydrop, _, _ = _render_half_pkg(
        cam, gaussians, renderFunc, renderArgs, env_map, args)
    depth, intensity_sh, raydrop, geom = ring_raydrop_maps(
        args, cam, depth, intensity_sh, raydrop, depth[[1]], env_map)
    device = depth.device

    drop = geom.slot_valid.to(device) < 0.5
    dead = getattr(cam, "ring_dead_mask", None)
    if dead is not None and bool(getattr(args, "blind_dead_mask", False)):
        drop = drop | (dead.to(device) > 0.5)
    from utils.blind_dead_mask import force_dead_drop
    depth, intensity_sh, raydrop = force_dead_drop(depth, intensity_sh, raydrop, drop.float())

    body_depth = getattr(cam, "ring_body_depth", None)
    if getattr(args, "ego_body_infer_replace", True) and body_depth is not None:
        from utils.ego_body_mask import paste_gt_ego_body_rv, _dilate_mask
        clear = body_depth > 0
        ego = getattr(cam, "ring_ego_mask", None)
        if ego is not None:
            clear = clear | (ego > 0.5)
        dilate = int(getattr(args, "ego_body_replace_dilate", 2) or 0)
        clear = _dilate_mask(clear, dilate) * geom.slot_valid
        depth, intensity_sh, raydrop = paste_gt_ego_body_rv(
            depth, intensity_sh, raydrop, body_depth, cam.ring_body_intensity, clear)

    return (depth, intensity_sh, raydrop,
            cam.ring_depth.to(device), cam.ring_intensity.to(device), geom)
