#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# Adapted for GACRT dataset format.
#
import json
from utils.system_utils import save_ply
import os
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf


def lidar_list_to_video(tensor_list, output_path, fps=10, scale_factor=1):
    h, w = tensor_list[0].shape[1], tensor_list[0].shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_path, '3d.mp4'), fourcc, fps,
                                   (w * scale_factor, h * scale_factor))
    to_pil_image = transforms.ToPILImage()

    for idx, tensor in tqdm(enumerate(tensor_list)):
        image = to_pil_image(tensor)
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (w * scale_factor, h * scale_factor))
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Video saved in {os.path.join(output_path, '3d.mp4')}")


def visualize_gacrt_ply(ply_path, ego2global, scale_factor=1.0,
                         img_size=(2086, 1248), depth_range=(-35, 40)):
    """将 GACRT PLY 点云投影到虚拟相机视角进行可视化。

    Args:
        ply_path: PLY文件路径
        ego2global: (4,4) ego到世界的变换矩阵 (用于选择合适视角)
        scale_factor: 训练时的缩放因子
        img_size: (W, H) 渲染图像尺寸
        depth_range: (min, max) 深度范围用于颜色映射
    Returns:
        RGB tensor [3, H, W]
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points) / scale_factor

    # GACRT 坐标系: ego frame 下 x=前, y=左, z=上
    # 转换到 KITTI-360 类似的坐标系以便可视化
    # KITTI: x=右, y=下, z=前 → 需要从 GACRT 的 x=前,y=左,z=上 转换
    # 使用 ego2global 的旋转来确定车辆前进方向
    R_world = ego2global[:3, :3]
    T_world = ego2global[:3, 3]

    # 将点云从世界坐标转到 ego 坐标系 (车辆坐标系)
    # ego2global: ego → world, 所以 inv(ego2global): world → ego
    ego2global_inv = np.linalg.inv(ego2global)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
    points_ego = (ego2global_inv @ points_homo.T).T[:, :3]

    # 深度（前方距离）
    z = np.sqrt(points_ego[:, 0] ** 2 + points_ego[:, 1] ** 2)
    z_min, z_max = depth_range
    color = np.clip((z - z_min) / (z_max - z_min), 0, 1)

    # 设置虚拟相机：位于 ego 原点，看向 x 正方向（前方）
    # 相机内参
    fx, fy = 1080.8, 1080.8
    W, H = img_size

    # 从 ego 坐标系投影到相机平面
    # ego: x=前, y=左, z=上 → camera: u=x方向(右为正), v=z方向(上为正)
    # 相机坐标系: 相机看向 x 正方向
    # u = -y (左→右翻转), v = z (高度), depth = x (前方距离)
    u = -points_ego[:, 1]  # 水平像素坐标 (右为正)
    v = points_ego[:, 2]   # 垂直像素坐标 (上为正)
    depth_cam = points_ego[:, 0]  # 深度（前方距离）

    # 只保留前方点
    mask_front = depth_cam > 0
    u = u[mask_front]
    v = v[mask_front]
    depth_cam = depth_cam[mask_front]
    color = color[mask_front]

    # 投影到像素坐标
    pixel_x = fx * u / depth_cam + W / 2
    pixel_y = -fy * v / depth_cam + H / 2  # 翻转 y (图像上方=高处)

    # 过滤在图像范围内的点
    mask_visible = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
    pixel_x = pixel_x[mask_visible]
    pixel_y = pixel_y[mask_visible]
    color = color[mask_visible]

    # 绘制
    plt.figure(facecolor='black', figsize=(W / 100.0, H / 100.0))
    plt.scatter(pixel_x, pixel_y, c=color, cmap='turbo', alpha=1, s=5, marker='.', edgecolors='none')
    plt.xlim([0, W])
    plt.ylim([0, H])
    plt.axis('off')
    plt.savefig('/tmp/gacrt_vis.png', bbox_inches='tight', pad_inches=0, dpi=130)
    plt.close()

    transform = transforms.ToTensor()
    lidar_image = Image.open('/tmp/gacrt_vis.png')
    return transform(lidar_image)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Visualize GACRT LiDAR reconstruction")
    parser.add_argument("--seq", type=str, required=True, help="GACRT sequence ID")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (default: eval_output/gacrt_reconstruction/{seq})")
    parser.add_argument("--config", type=str, required=True, help="Config yaml file")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"eval_output/gacrt_reconstruction/{args.seq}"

    # Load config for scale_factor and transforms
    base_conf = OmegaConf.load("configs/base.yaml")
    scene_conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(base_conf, scene_conf)
    conf.source_path = "./data"
    with open(os.path.join(args.model_path, "scale_factor.txt"), 'r') as f:
        scale_factor = float(f.read())

    # Load pkl for ego2global transforms
    from scene.gacrt_loader import load_pkl_compat
    pkl_path = getattr(conf, 'pkl_path', None)
    if pkl_path is None:
        import glob
        pkl_candidates = glob.glob(os.path.join(conf.source_path, f"{args.seq}.pkl"))
        if not pkl_candidates:
            raise FileNotFoundError(f"Cannot find pkl file for {args.seq}")
        pkl_path = pkl_candidates[0]

    pkl_data = load_pkl_compat(pkl_path)

    # Find all PLY files in refine render directory
    refine_dirs = []
    for mode in ["train", "test"]:
        dir_path = os.path.join(args.model_path, f"eval/{mode}_refine_render")
        if os.path.exists(dir_path):
            refine_dirs.append((mode, dir_path))

    if not refine_dirs:
        print("No refine render directories found. Run training first.")
        exit(1)

    # Collect all PLY files
    all_ply_frames = {}  # frame_id → (ply_path, mode)

    for mode, dir_path in refine_dirs:
        for fname in sorted(os.listdir(dir_path)):
            if fname.endswith('.ply') and "gt_depth" not in fname:
                frame_id = int(fname.replace('.ply', ''))
                all_ply_frames[frame_id] = (os.path.join(dir_path, fname), mode)

    frame_ids = sorted(all_ply_frames.keys())
    print(f"Found {len(frame_ids)} PLY frames")

    # Map colmap_id (frame_id from PLY naming) back to keyframe index
    # PLY files are named by colmap_id which corresponds to the camera index
    # For GACRT with 2 cameras per frame (forward/backward):
    # colmap_id = frame_idx (forward) or frame_idx + frames (backward)
    # We use forward camera's ego2global for visualization
    total_keyframes = len(pkl_data)

    # Load JSON to get frame count
    json_path = os.path.join(conf.source_path, args.seq, f"transforms_{args.seq}_all.json")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    total_frames = len(json_data["frames"])
    cam_num = getattr(conf, 'cam_num', 2)

    video_frames = []
    for frame_id in tqdm(frame_ids[:50]):  # Limit to 50 frames for speed
        ply_path, mode = all_ply_frames[frame_id]

        # Determine which ego2global to use
        # colmap_id for forward camera = frame_idx
        # frame_idx = colmap_id // cam_num (for forward cam)
        frame_idx = frame_id // cam_num
        if frame_idx < total_keyframes:
            ego2global = pkl_data[frame_idx]["ego2global"]
        else:
            # Use closest keyframe
            ego2global = pkl_data[min(frame_idx, total_keyframes - 1)]["ego2global"]

        rgb_tensor = visualize_gacrt_ply(
            ply_path, ego2global, scale_factor=scale_factor,
            img_size=(2086, 1248), depth_range=(-35, 40)
        )
        video_frames.append(rgb_tensor)

    # Clean up temp file
    if os.path.exists('/tmp/gacrt_vis.png'):
        os.remove('/tmp/gacrt_vis.png')

    output_path = os.path.join(args.model_path, "eval/others")
    os.makedirs(output_path, exist_ok=True)

    if video_frames:
        lidar_list_to_video(video_frames, output_path, fps=args.fps)
        print(f"Video with {len(video_frames)} frames saved to {output_path}/3d.mp4")
    else:
        print("No frames to visualize")