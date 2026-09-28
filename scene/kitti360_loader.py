#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from torchvision.utils import save_image
import torch.nn.functional as F
import json
from matplotlib import cm
from utils.system_utils import save_ply


def range_to_ply(depth, filename, vfov=(-31.96, 10.67), hfov=(-90, 90)):
    panorama_height, panorama_width = depth.shape[-2:]

    theta, phi = torch.meshgrid(torch.arange(panorama_height, device='cuda'),
                                torch.arange(panorama_width, device='cuda'), indexing="ij")

    vertical_degree_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

    horizontal_degree_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)
    points_xyz = directions * depth

    save_ply(points_xyz.reshape(3, -1).permute(1, 0), filename)

    return


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_scale_factor=True):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    if fix_scale_factor:
        scale_factor = 1 / 10
    else:
        # Just make sure it's it in the [-1, 1]^3 cube
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


def readKitti360Info(args):
    path = args.source_path
    eval = args.eval
    num_pts = args.num_pts
    time_duration = args.time_duration
    debug_cuda = args.debug_cuda

    assert args.vfov is not None and args.hfov is not None

    sequence_name = "2013_05_28_drive_0000_sync"
    sequence_id = args.sequence_id

    # static
    if sequence_id == "1538":
        print("Using sqequence 1538-1601")
        s_frame_id = 1538
        e_frame_id = 1601  # Inclusive
        val_frame_ids = [1551, 1564, 1577, 1590]
    elif sequence_id == "1728":
        print("Using sqequence 1728-1791")
        s_frame_id = 1728
        e_frame_id = 1791  # Inclusive
        val_frame_ids = [1741, 1754, 1767, 1780]
    elif sequence_id == "1908":
        print("Using sqequence 1908-1971")
        s_frame_id = 1908
        e_frame_id = 1971  # Inclusive
        val_frame_ids = [1921, 1934, 1947, 1960]
    elif sequence_id == "3353":
        print("Using sqequence 3353-3416")
        s_frame_id = 3353
        e_frame_id = 3416  # Inclusive
        val_frame_ids = [3366, 3379, 3392, 3405]
    # dynamic
    elif sequence_id == "2350":
        s_frame_id = 2350
        e_frame_id = 2400  # Inclusive
        val_frame_ids = [2360, 2370, 2380, 2390]
    elif sequence_id == "4950":
        s_frame_id = 4950
        e_frame_id = 5000  # Inclusive
        val_frame_ids = [4960, 4970, 4980, 4990]
    elif sequence_id == "8120":
        s_frame_id = 8120
        e_frame_id = 8170  # Inclusive
        val_frame_ids = [8130, 8140, 8150, 8160]
    elif sequence_id == "10200":
        s_frame_id = 10200
        e_frame_id = 10250  # Inclusive
        val_frame_ids = [10210, 10220, 10230, 10240]
    elif sequence_id == "10750":
        s_frame_id = 10750
        e_frame_id = 10800  # Inclusive
        val_frame_ids = [10760, 10770, 10780, 10790]
    elif sequence_id == "11400":
        s_frame_id = 11400
        e_frame_id = 11450  # Inclusive
        val_frame_ids = [11410, 11420, 11430, 11440]
    else:
        raise ValueError(f"Invalid sequence id: {sequence_id}")

    with open(os.path.join(path, f"{sequence_id}", f"transforms_{sequence_id}_all.json"), "r") as file:
        data = json.load(file)

    poses = data["frames"]

    frames = e_frame_id + 1 - s_frame_id
    args.frames = frames
    lidar_dir = os.path.join(path, "KITTI-360", "data_3d_raw", sequence_name, "velodyne_points", "data")

    point_list = []
    points_time = []
    cam_infos = []

    for frame_idx in tqdm(range(frames), desc="Reading Kitti360Info"):
        lidar_idx = frame_idx + s_frame_id
        points = np.fromfile(os.path.join(lidar_dir, "%010d.bin" % lidar_idx), dtype=np.float32).reshape((-1, 4))
        intensity = points[:, 3]
        points = points[:, :3]

        # 把自车的lidar点去掉
        condition = (np.linalg.norm(points, axis=1) > 2.5)  # & (intensity > 0)
        indices = np.where(condition)
        points = points[indices]
        intensity = intensity[indices]

        lidar2globals = np.array(poses[frame_idx]["lidar2world"])

        points_homo = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
        points = (points_homo @ lidar2globals.T)[:, :3]
        point_list.append(points)

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (frames - 1)
        point_time = np.full_like(points[:, :1], timestamp)
        points_time.append(point_time)

        idx = frame_idx
        w2l = np.array([0, -1, 0, 0,
                        0, 0, -1, 0,
                        1, 0, 0, 0,
                        0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(lidar2globals)
        R = np.transpose(w2l[:3, :3])
        T = w2l[:3, 3]
        points_cam = points @ R + T

        # 前180度
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T,
                                    timestamp=timestamp, pointcloud_camera=points_cam, intensity=intensity,
                                    towards='forward', sensor_id=0, frame_idx=frame_idx))

        # 后180度
        R_back = R @ np.array([-1, 0, 0,
                               0, 1, 0,
                               0, 0, -1]).reshape(3, 3)
        T_back = T * np.array([-1, 1, -1])
        points_cam_back = points @ R_back + T_back
        cam_infos.append(CameraInfo(uid=idx + frames, R=R_back, T=T_back,
                                    timestamp=timestamp, pointcloud_camera=points_cam_back, intensity=intensity,
                                    towards='backward', sensor_id=0, frame_idx=frame_idx))

        if debug_cuda and frame_idx >= 15:
            break

    pointcloud = np.concatenate(point_list, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)

    num_pts = min(num_pts, pointcloud.shape[0])
    indices = np.random.choice(pointcloud.shape[0], num_pts, replace=False)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))

    if not args.test_only:
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, args.dynamic)
        np.savez(os.path.join(args.model_path, 'transform_poses_pca.npz'), transform=transform, scale_factor=scale_factor)
        c2ws = pad_poses(c2ws)
    else:
        data = np.load(os.path.join(args.model_path, 'transform_poses_pca.npz'))
        transform = data['transform']
        scale_factor = data['scale_factor'].item()
        c2ws = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2ws)
        c2ws[:, :3, 3] *= scale_factor

    for idx, cam_info in enumerate(cam_infos):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor

    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    args.scale_factor = float(scale_factor)

    mod = args.cam_num

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // mod + s_frame_id) not in val_frame_ids]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // mod + s_frame_id) in val_frame_ids]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // mod + s_frame_id) in val_frame_ids]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1

    ply_path = os.path.join(args.model_path, "points3d.ply")
    if not args.test_only:
        rgbs = np.random.random((pointcloud.shape[0], 3)) * 255.0
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    time_interval = (time_duration[1] - time_duration[0]) / (frames - 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval)

    return scene_info
