#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
import kornia
from torchvision.utils import save_image


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, vfov=None, hfov=None, uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", timestamp=0.0,
                 resolution=None, image_path=None,
                 pts_depth=None, pts_intensity=None, towards=None,
                 od_loss_mask=None, pts_depth_input=None,
                 sensor_id=0, frame_idx=None, ego_body_mask=None,
                 ego_body_points_cam=None, ego_body_intensity=None,
                 blind_dead_mask=None, blind_ignore_mask=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.vfov = vfov
        self.hfov = hfov
        self.resolution = resolution
        self.image_path = image_path
        self.towards = towards
        self.sensor_id = int(sensor_id or 0)
        self.frame_idx = None if frame_idx is None else int(frame_idx)
        # Rigid body returns in this camera frame; pasted at inference.
        self.ego_body_points_cam = ego_body_points_cam
        self.ego_body_intensity = ego_body_intensity

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]

        # Store GT depth/intensity on CPU to save GPU memory; move to GPU on access via .cuda()
        self.pts_depth = pts_depth.cpu() if pts_depth is not None else pts_depth
        self.pts_intensity = pts_intensity.cpu() if pts_intensity is not None else pts_intensity
        self.od_loss_mask = od_loss_mask.cpu() if od_loss_mask is not None else od_loss_mask
        self.ego_body_mask = ego_body_mask.cpu() if ego_body_mask is not None else ego_body_mask
        self.blind_dead_mask = blind_dead_mask.cpu() if blind_dead_mask is not None else None
        self.blind_ignore_mask = blind_ignore_mask.cpu() if blind_ignore_mask is not None else None
        self.pts_depth_input = pts_depth_input.cpu() if pts_depth_input is not None else pts_depth_input

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = torch.eye(4).cuda()  # no use
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.grid = kornia.utils.create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device='cuda')[0]

    def get_world_directions(self, train=False):
        u, v = self.grid.unbind(-1)
        if train:
            directions = torch.stack([(u - self.cx + torch.rand_like(u)) / self.fx,
                                      (v - self.cy + torch.rand_like(v)) / self.fy,
                                      torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u - self.cx + 0.5) / self.fx,
                                      (v - self.cy + 0.5) / self.fy,
                                      torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    def get_world_directions_panorama(self, train=False):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        if train:
            theta = theta + torch.rand_like(theta.float()) - 0.5
            phi = phi + torch.rand_like(phi.float()) - 0.5

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.image_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.image_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height,
                                                                            self.image_width)
        return directions

    def get_local_directions_panorama(self, train=False):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.image_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.image_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        assert self.towards is not None
        if self.towards == 'forward':
            directions = torch.stack([dx, dy, dz], dim=0)
        else:
            directions = torch.stack([-dx, dy, -dz], dim=0)
        directions = F.normalize(directions, dim=0)
        return directions
