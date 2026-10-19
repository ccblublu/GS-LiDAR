#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import torch
import cv2
from scene.cameras import Camera
import numpy as np
from scene.scene_utils import CameraInfo
from tqdm import tqdm
from torchvision.utils import save_image


def project_points_to_rv(points, intensity, h, w, vfov, hfov):
    """Project camera-frame points onto a panoramic range/intensity map."""
    pts_depth = np.zeros([1, h, w])
    pts_intensity = np.zeros([1, h, w])
    if points is None or len(points) == 0:
        return pts_depth, pts_intensity
    if intensity is None:
        intensity = np.ones(points.shape[0], dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    phi = np.arctan2(x, z)
    theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
    VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
    HFOV_max = hfov[1] * np.pi / 180
    HFOV_min = hfov[0] * np.pi / 180

    theta = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
    phi = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)
    uvz = np.stack((theta, phi, r, intensity), 1)

    uvz = uvz[uvz[:, 0] >= -0.5]
    uvz = uvz[uvz[:, 0] < h - 0.5]
    uvz = uvz[uvz[:, 1] >= -0.5]
    uvz = uvz[uvz[:, 1] < w - 0.5]
    uv = np.around(uvz[:, :2]).astype(int)

    for i in range(uv.shape[0]):
        row, col = uv[i]
        if pts_depth[0, row, col] == 0 or uvz[i, 2] < pts_depth[0, row, col]:
            pts_depth[0, row, col] = uvz[i, 2]
            pts_intensity[0, row, col] = uvz[i, 3]
    return pts_depth, pts_intensity


def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    orig_h, orig_w = args.hw

    if args.resolution == -1:
        global_down = 1
    else:
        global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    vfov = args.vfov
    hfov = args.hfov
    w = resolution[0]
    h = resolution[1]
    if cam_info.pointcloud_camera is not None:
        intensity = cam_info.intensity
        if intensity is None:
            intensity = np.ones_like(cam_info.pointcloud_camera)[:, 0]

        pts_depth, pts_intensity = project_points_to_rv(
            cam_info.pointcloud_camera, intensity, h, w, vfov, hfov)
        pts_depth = torch.from_numpy(pts_depth).float().cuda()
        pts_intensity = torch.from_numpy(pts_intensity).float().cuda()
    else:
        pts_depth = None
        pts_intensity = None

    pts_depth_input = None
    if cam_info.pointcloud_camera_input is not None:
        intensity_in = cam_info.intensity_input
        if intensity_in is None:
            intensity_in = np.ones_like(cam_info.pointcloud_camera_input)[:, 0]
        depth_in, _ = project_points_to_rv(
            cam_info.pointcloud_camera_input, intensity_in, h, w, vfov, hfov)
        pts_depth_input = torch.from_numpy(depth_in).float().cuda()

    od_loss_mask = None
    if cam_info.od_loss_mask is not None:
        od_loss_mask = torch.from_numpy(cam_info.od_loss_mask).float()
        if od_loss_mask.shape[-2:] != (h, w):
            od_loss_mask = torch.nn.functional.interpolate(
                od_loss_mask.unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0)
        # Zero remaining silhouette pixels (expanded bbox / projection leftovers).
        if pts_depth is not None:
            keep = od_loss_mask.to(pts_depth.device) > 0.5
            pts_depth = pts_depth * keep
            pts_intensity = pts_intensity * keep
        if pts_depth_input is not None:
            keep = od_loss_mask.to(pts_depth_input.device) > 0.5
            pts_depth_input = pts_depth_input * keep

    ego_body_mask = getattr(cam_info, "ego_body_mask", None)
    if ego_body_mask is not None:
        ego_body_mask = torch.from_numpy(np.asarray(ego_body_mask)).float()
        if ego_body_mask.ndim == 2:
            ego_body_mask = ego_body_mask.unsqueeze(0)
        if ego_body_mask.shape[-2:] != (h, w):
            ego_body_mask = torch.nn.functional.interpolate(
                ego_body_mask.unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0)

    ego_body_points_cam = getattr(cam_info, "ego_body_points_cam", None)
    if ego_body_points_cam is not None:
        ego_body_points_cam = np.asarray(ego_body_points_cam, dtype=np.float32)
    ego_body_intensity = getattr(cam_info, "ego_body_intensity", None)
    if ego_body_intensity is not None:
        ego_body_intensity = np.asarray(ego_body_intensity, dtype=np.float32)

    from utils.blind_dead_mask import resize_dead_mask
    blind_dead_mask = resize_dead_mask(
        getattr(cam_info, "blind_dead_mask", None), h, w, any_pixel=False)
    blind_ignore_mask = resize_dead_mask(
        getattr(cam_info, "blind_ignore_mask", None), h, w, any_pixel=True)

    ring_gt = {}
    from utils.ring_rv import ring_rv_enabled, build_camera_ring_gt
    if ring_rv_enabled(args):
        ring_gt = build_camera_ring_gt(args, cam_info, h, w)

    return Camera(
        colmap_id=cam_info.uid,
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        vfov=vfov,
        hfov=hfov,
        data_device=args.data_device,
        timestamp=cam_info.timestamp,
        resolution=resolution,
        pts_depth=pts_depth,
        pts_intensity=pts_intensity,
        towards=cam_info.towards,
        od_loss_mask=od_loss_mask,
        pts_depth_input=pts_depth_input,
        sensor_id=getattr(cam_info, "sensor_id", 0),
        frame_idx=getattr(cam_info, "frame_idx", None),
        ego_body_mask=ego_body_mask,
        ego_body_points_cam=ego_body_points_cam,
        ego_body_intensity=ego_body_intensity,
        blind_dead_mask=blind_dead_mask,
        blind_ignore_mask=blind_ignore_mask,
        ring_gt=ring_gt,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list
