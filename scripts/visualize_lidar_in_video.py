#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
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


def lidar_list_to_video(tensor_list, output_path, fps=30, scale_factor=1):
    # 确保每个 Tensor 是 [C, H, W]，通常 C = 3 (RGB)
    h, w = tensor_list[0].shape[1], tensor_list[0].shape[2]  # 获取高度和宽度

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    video_writer = cv2.VideoWriter(os.path.join(output_path, '3d.mp4'), fourcc, fps, (w * scale_factor, h * scale_factor))
    to_pil_image = transforms.ToPILImage()

    for idx, tensor in tqdm(enumerate(tensor_list)):
        # 确保 tensor 是 [C, H, W]，其中 C = 3
        # 转换为 PIL 图像再转为 NumPy
        image = to_pil_image(tensor)
        frame = np.array(image)

        # OpenCV 使用 BGR 格式，需要转换
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (w * scale_factor, h * scale_factor))

        # 写入帧
        video_writer.write(frame_bgr)

    # 释放视频写入器
    video_writer.release()
    print(f"Video saved in {os.path.join(output_path, '3d.mp4')}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=str,
        default="1908",
        help="Sequence to use.",
    )
    args = parser.parse_args()

    root_path = f"eval_output/gacrt_reconstruction/{args.seq}"
    with open(os.path.join(root_path, "scale_factor.txt"), 'r') as file:
        data = file.read()
        scale_factor = float(data)

    transform = transforms.ToTensor()
    video_frame = []
    lidar_frame = []

    for i in tqdm(range(51)):
        if os.path.exists(os.path.join(root_path, "eval/train_refine_render", f"{i:03d}.png")):
            mode = "train"
        elif os.path.exists(os.path.join(root_path, "eval/test_refine_render", f"{i:03d}.png")):
            mode = "test"
        else:
            print(i)
            raise FileNotFoundError

        pcd = o3d.io.read_point_cloud(os.path.join(root_path, f"eval/{mode}_refine_render", f"{i:03d}.ply"))
        points = np.asarray(pcd.points) / scale_factor
        points = points[:, [2, 0, 1]]
        points[:, [1, 2]] *= -1

        z = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        z_min = -35
        z_max = 40
        color = np.clip((z - z_min) / (z_max - z_min), 0, 1)

        R = np.array([-0.00991791, -0.999895, -0.0106093,
                      0.491445, -0.0141142, 0.870794,
                      -0.870852, 0.00342247, 0.491534]).reshape(3, 3)
        T = np.array([9.39696, -0.0206351, -3.84181])
        K = np.array([1080.8, 0, 1043,
                      0, 1080.8, 624,
                      0, 0, 1]).reshape(3, 3)

        coordinate_transform = np.array([1, -1, -1])

        W, H = (2086, 1248)

        points = ((points + T) @ R.T * coordinate_transform) @ K.T
        color = color[points[:, 2] > 0]
        points = points[points[:, 2] > 0]
        points[:, 0:2] /= points[:, [2]]

        color = color[points[:, 0] >= 0]
        points = points[points[:, 0] >= 0]
        color = color[points[:, 0] < W]
        points = points[points[:, 0] < W]
        color = color[points[:, 1] >= 0]
        points = points[points[:, 1] >= 0]
        color = color[points[:, 1] < H]
        points = points[points[:, 1] < H]

        plt.figure(facecolor='black', figsize=(W / 100.0, H / 100.0))
        plt.scatter(points[:, 0], H - points[:, 1], c=color, cmap='viridis', alpha=1, s=20, marker='.', edgecolors='none')
        plt.xlim([0, W])
        plt.ylim([0, H])
        plt.axis('off')
        plt.savefig('tmp.png', bbox_inches='tight', pad_inches=0, dpi=130)
        plt.close()

        lidar_image = Image.open('tmp.png')
        lidar_frame.append(transform(lidar_image))

    os.remove('tmp.png')
    output_path = os.path.join(root_path, "eval/others")
    os.makedirs(output_path, exist_ok=True)
    lidar_list_to_video(lidar_frame, output_path, fps=10)
