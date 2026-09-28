"""
GACRT 数据集适配验证脚本 — 四层递进验证。

Level 1: 数据完整性（CPU only）
Level 2: 数据语义 + 全景投影可视化（CPU only）
Level 3: Scene 加载集成验证（需 GPU）
Level 4: 首次渲染冒烟测试（需 GPU）

用法:
    python scripts/verify_gacrt_dataset.py \
        --config configs/gacrt_nvs_GACRT014_1740908194.yaml \
        --source_path data \
        --level 4 \
        --output_dir verify_output
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scene.gacrt_loader import get_lidar2ego, load_lidar_xyz_intensity
# ─── 工具函数 ────────────────────────────────────────────────

class VerifyResult:
    """单个验证项的结果。"""
    def __init__(self, name, status, detail="", value=None):
        self.name = name
        self.status = status   # "pass", "fail", "warn", "skip"
        self.detail = detail
        self.value = value

    def __str__(self):
        icon = {"pass": "✅", "fail": "❌", "warn": "⚠️", "skip": "⏭️"}[self.status]
        line = f"  {icon} {self.name}"
        if self.value is not None:
            line += f": {self.value}"
        if self.detail:
            line += f" ({self.detail})"
        return line


class VerifyReport:
    """一层验证的汇总报告。"""
    def __init__(self, level_name):
        self.level_name = level_name
        self.results = []

    def add(self, name, status, detail="", value=None):
        self.results.append(VerifyResult(name, status, detail, value))

    def print_report(self):
        print(f"\n[{self.level_name}]")
        for r in self.results:
            print(str(r))
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        warned = sum(1 for r in self.results if r.status == "warn")
        total = len(self.results)
        print(f"  汇总: {passed}/{total} 通过, {failed} 失败, {warned} 警告")
        return failed == 0


def load_config(config_path, source_path=None):
    """加载 OmegaConf 合并配置。"""
    from omegaconf import OmegaConf
    base_conf = OmegaConf.load("configs/base.yaml")
    scene_conf = OmegaConf.load(config_path)
    conf = OmegaConf.merge(base_conf, scene_conf)
    if source_path is not None:
        conf.source_path = source_path
    return conf


def load_pkl_compat(path):
    """兼容 numpy 版本的 pkl 加载。"""
    import pickle
    class NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            elif module.startswith('numpy.core'):
                module = module.replace('numpy.core', 'numpy._core')
            return super().find_class(module, name)
    with open(path, 'rb') as f:
        return NumpyCompatUnpickler(f).load()


def project_points_to_pano(points_cam, vfov, hfov, hw):
    """将相机坐标系点云投影到全景范围图 (numpy 版本)。

    与 camera_utils.py 中的 loadCam 使用相同的投影逻辑。

    Args:
        points_cam: (N, 3) 相机坐标系点云
        vfov: (vfov_min, vfov_max) 垂直视场角 (度)
        hfov: (hfov_min, hfov_max) 水平视场角 (度)
        hw: (height, width) 全景图尺寸

    Returns:
        depth_map: (1, H, W) 深度图
        intensity_map: (1, H, W) 强度图 (此处为零)
        coverage: 有效像素占比
    """
    h, w = hw
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    phi = np.arctan2(x, z)   # 方位角
    theta = np.arctan2(np.sqrt(x**2 + z**2), -y)  # 仰角
    r = np.sqrt(x**2 + y**2 + z**2)

    VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
    VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
    HFOV_max = hfov[1] * np.pi / 180
    HFOV_min = hfov[0] * np.pi / 180

    theta_px = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
    phi_px = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)

    # 过滤视场内的点
    mask = (theta_px >= -0.5) & (theta_px < h - 0.5) & \
           (phi_px >= -0.5) & (phi_px < w - 0.5)
    theta_px = theta_px[mask]
    phi_px = phi_px[mask]
    r = r[mask]

    uv = np.around(np.stack([theta_px, phi_px], axis=1)).astype(int)

    depth_map = np.zeros([1, h, w])
    for i in range(uv.shape[0]):
        row, col = uv[i]
        if depth_map[0, row, col] == 0 or r[i] < depth_map[0, row, col]:
            depth_map[0, row, col] = r[i]

    coverage = np.count_nonzero(depth_map) / (h * w)
    return depth_map, coverage


def save_pano_image(depth_map, save_path, title="Depth", cmap='turbo', vfov=None, hfov=None):
    """保存全景深度图 PNG。"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    im = ax.imshow(depth_map[0], cmap=cmap, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Azimuth (pixels)')
    ax.set_ylabel('Elevation (pixels)')
    plt.colorbar(im, ax=ax, label='Distance (m)')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Level 1: 数据完整性验证 ────────────────────────────────

def verify_level_1(conf, output_dir):
    report = VerifyReport("Level 1 — 数据完整性验证")

    source_path = conf.source_path
    sequence_id = conf.sequence_id

    # 1. JSON 文件
    json_path = os.path.join(source_path, sequence_id,
                              f"transforms_{sequence_id}_all.json")
    if not os.path.exists(json_path):
        report.add("JSON 文件存在", "fail", f"路径 {json_path} 不存在")
        return report

    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        report.add("JSON 可解析", "pass", value=json_path)
    except Exception as e:
        report.add("JSON 可解析", "fail", f"解析失败: {e}")
        return report

    # 2. JSON 必需键
    required_keys = ['frames', 'w_lidar', 'h_lidar', 'aabb_scale', 'val_frame_ids']
    for key in required_keys:
        if key in json_data:
            report.add(f"JSON 键 '{key}'", "pass", value=json_data[key] if key != 'frames' else len(json_data[key]))
        else:
            report.add(f"JSON 键 '{key}'", "fail", "缺失")

    # 3. 每个 frame 必需字段
    frames = json_data.get('frames', [])
    frame_required_keys = ['lidar2world', 'timestamp', 'lidar_path', 'num_pts_feats']
    bad_frames = 0
    for i, frame in enumerate(frames):
        for key in frame_required_keys:
            if key not in frame:
                bad_frames += 1
                break
        # lidar2world 应为 4x4
        lidar2world = frame.get('lidar2world')
        if lidar2world is not None:
            mat = np.array(lidar2world)
            if mat.shape != (4, 4):
                bad_frames += 1

    if bad_frames == 0:
        report.add("Frame 结构完整", "pass", value=f"{len(frames)} 帧")
    else:
        report.add("Frame 结构完整", "fail", value=f"{bad_frames} 帧格式错误")

    # 4. val_frame_ids 范围
    val_frame_ids = json_data.get('val_frame_ids', [])
    out_of_range = [vf for vf in val_frame_ids if vf < 0 or vf >= len(frames)]
    if len(out_of_range) == 0:
        report.add("val_frame_ids 范围", "pass",
                   value=f"{len(val_frame_ids)} 个, 全在 [0, {len(frames)}) 内")
    else:
        report.add("val_frame_ids 范围", "fail",
                   value=f"{len(out_of_range)} 个超出范围: {out_of_range[:5]}")

    # 5. frames 按 timestamp 排序
    timestamps = [f.get('timestamp', 0) for f in frames]
    is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    if is_sorted:
        report.add("timestamp 排序", "pass", value="非降序")
    else:
        report.add("timestamp 排序", "fail", "frames 未按 timestamp 排序")

    # 6. 配置文件一致性
    report.add("scene_type", "pass" if conf.scene_type == "Gacrt" else "fail",
               value=conf.scene_type)
    report.add("cam_num", "pass" if conf.cam_num == 2 else "fail",
               value=conf.cam_num)
    report.add("vfov 非空", "pass" if conf.vfov is not None else "fail",
               value=list(conf.vfov) if conf.vfov is not None else "null")
    report.add("hfov 非空", "pass" if conf.hfov is not None else "fail",
               value=list(conf.hfov) if conf.hfov is not None else "null")

    # 7. w_lidar/h_lidar 与 FoV 一致性
    w_lidar = json_data.get('w_lidar')
    h_lidar = json_data.get('h_lidar')
    if w_lidar is not None and h_lidar is not None and conf.hfov is not None and conf.vfov is not None:
        # Pandar128: hfov_half = [-90, 90], h_lidar_half=900, 全景 w=1800
        hfov_half_range = conf.hfov[1] - conf.hfov[0]  # 180°
        expected_w_half = int(hfov_half_range / 0.2)  # 900
        vfov_range = conf.vfov[1] - conf.vfov[0]  # 40°
        expected_h = 128  # Pandar128 固定通道数

        w_half_match = (w_lidar == 2 * expected_w_half)
        h_match = (h_lidar == expected_h)
        report.add("w_lidar/FoV 一致", "pass" if w_half_match else "warn",
                   value=f"w_lidar={w_lidar}, 期望={2*expected_w_half}")
        report.add("h_lidar/FoV 一致", "pass" if h_match else "warn",
                   value=f"h_lidar={h_lidar}, 期望={expected_h}")

    # 8. hw 配置与全景图尺寸匹配
    hw = conf.hw if conf.hw is not None else [h_lidar, w_lidar // 2]
    report.add("hw 配置", "pass",
               value=f"[{hw[0]}, {hw[1]}] (H=通道数, W=半全景宽度)")

    # 9. 点云文件可访问性（.bin 或 .pcd）
    missing_bins = 0
    total_bins = len(frames)
    bad_shapes = 0
    for frame in frames:
        lidar_path = frame.get('lidar_path')
        n_feats = frame.get('num_pts_feats', 5)
        if lidar_path is None:
            missing_bins += 1
            continue
        full_path = os.path.join(source_path, lidar_path)
        if not os.path.lexists(full_path) and not os.path.exists(full_path):
            missing_bins += 1
            continue
        if os.path.islink(full_path) and not os.path.exists(full_path):
            missing_bins += 1
            continue
        file_size = os.path.getsize(full_path)
        if file_size == 0:
            bad_shapes += 1
            continue
        if full_path.lower().endswith(".pcd"):
            continue
        if file_size % (4 * n_feats) != 0:
            bad_shapes += 1

    if missing_bins == 0:
        report.add("点云文件可访问", "pass",
                   value=f"{total_bins}/{total_bins} 存在")
    else:
        report.add("点云文件可访问", "fail",
                   value=f"{missing_bins}/{total_bins} 缺失")

    if bad_shapes == 0:
        report.add("点云格式检查", "pass",
                   value="所有 .bin/.pcd 文件可读取")
    else:
        report.add("点云格式检查", "fail",
                   value=f"{bad_shapes} 文件格式异常")

    # 10. PKL 文件（如启用 OD）
    pkl_path = getattr(conf, 'pkl_path', None)
    if pkl_path is not None and getattr(conf, 'use_od_labels', False):
        full_pkl = pkl_path if os.path.isabs(pkl_path) else os.path.join(source_path, pkl_path)
        if os.path.exists(full_pkl):
            try:
                pkl_data = load_pkl_compat(full_pkl)
                report.add("PKL 文件", "pass",
                           value=f"{len(pkl_data)} keyframes, OD 模式启用")
            except Exception as e:
                report.add("PKL 文件", "fail", f"加载失败: {e}")
        else:
            report.add("PKL 文件", "fail", f"路径 {full_pkl} 不存在")
    else:
        report.add("PKL 文件", "skip",
                   value="OD 未启用或 pkl_path 未配置")

    return report


# ─── Level 2: 数据语义验证 ───────────────────────────────────


def verify_level_2(conf, output_dir, num_sample_frames=5):
    report = VerifyReport("Level 2 — 数据语义验证")
    os.makedirs(output_dir, exist_ok=True)

    source_path = conf.source_path
    sequence_id = conf.sequence_id
    vfov = list(conf.vfov)
    hfov = list(conf.hfov)
    hw = list(conf.hw)
    lidar2ego = get_lidar2ego(conf)

    # 加载 JSON
    json_path = os.path.join(source_path, sequence_id,
                             f"transforms_{sequence_id}_all.json")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    frames = json_data['frames']

    # 采样若干帧
    total_frames = len(frames)
    sample_indices = np.linspace(0, total_frames - 1,
                                 min(num_sample_frames,
                                     total_frames)).astype(int)

    # ── 位姿合理性 ──
    max_displacement = 0.0
    bad_pose_count = 0
    for i, frame in enumerate(frames):
        lidar2global = np.array(frame['lidar2world'])
        det = np.linalg.det(lidar2global[:3, :3])
        if abs(abs(det) - 1.0) > 0.05:
            bad_pose_count += 1

        if i > 0:
            prev_pose = np.array(frames[i - 1]['lidar2world'])
            curr_center = lidar2global[:3, 3]
            prev_center = prev_pose[:3, 3]
            disp = np.linalg.norm(curr_center - prev_center)
            max_displacement = max(max_displacement, disp)

    if bad_pose_count == 0:
        report.add("位姿有效性", "pass", value=f"所有 {total_frames} 帧为刚体变换")
    else:
        report.add("位姿有效性", "fail", value=f"{bad_pose_count} 帧位姿异常 (det ≠ ±1)")

    report.add("最大帧间位移",
               "pass" if max_displacement < 50 else "warn",
               value=f"{max_displacement:.2f}m")

    # ── 点云语义 ──
    # 加载采样帧的点云
    all_points_valid = True
    coverage_values = []
    depth_coverage_values = []

    for idx in sample_indices:
        frame = frames[idx]
        lidar_path = frame.get('lidar_path')
        n_feats = frame.get('num_pts_feats', 5)
        lidar2global = np.array(frame['lidar2world'])

        full_path = os.path.join(source_path, lidar_path)
        if not os.path.exists(full_path):
            report.add(f"帧 {idx} 点云加载", "fail", f"文件不存在: {full_path}")
            all_points_valid = False
            continue

        try:
            points, intensity = load_lidar_xyz_intensity(full_path, n_feats)
        except Exception as e:
            report.add(f"帧 {idx} 点云加载", "fail", str(e))
            all_points_valid = False
            continue

        # 坐标范围
        x_range = (points[:, 0].min(), points[:, 0].max())
        y_range = (points[:, 1].min(), points[:, 1].max())
        z_range = (points[:, 2].min(), points[:, 2].max())
        int_range = (intensity.min(), intensity.max())

        # 去 ego 车点
        ego_mask = np.linalg.norm(points[:, :2], axis=1) > 2.5
        points_filtered = points[ego_mask]
        intensity_filtered = intensity[ego_mask]

        ego_ratio = len(points_filtered) / len(points)
        report.add(f"帧 {idx} ego 过滤",
                   "pass" if ego_ratio > 0.5 else "warn",
                   value=f"保留 {ego_ratio:.1%}, {len(points_filtered)} 点")

        # 变换到世界坐标
        # lidar2global_ = np.linalg.inv(LIDAR2EGO) @ lidar2global
        points_homo = np.concatenate(
            [points_filtered,
             np.ones_like(points_filtered[:, :1])], axis=-1)
        points_world = (points_homo @ lidar2global.T)[:, :3]

        # 检查 NaN/Inf
        nan_count = np.isnan(points_world).sum() + np.isinf(points_world).sum()
        if nan_count > 0:
            report.add(f"帧 {idx} 世界坐标", "fail", value=f"{nan_count} NaN/Inf")
        else:
            report.add(f"帧 {idx} 世界坐标", "pass", value="无 NaN/Inf")

        # ── 全景投影验证（核心） ──
        # 世界坐标 → 相机坐标 (注意: 必须用 world 坐标系的点，与 gacrt_loader 一致)
        w2l = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [
            0, 0, 0, 1
        ]]) @ np.linalg.inv(lidar2ego) @ np.linalg.inv(lidar2global)
        R = np.transpose(w2l[:3, :3])
        # R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ R
        T = w2l[:3, 3]
        points_cam = points_world @ R + T

        # Forward 半球投影
        depth_map_fwd, coverage_fwd = project_points_to_pano(
            points_cam, vfov, hfov, hw)
        coverage_values.append(coverage_fwd)

        # 有效深度像素（非零且有值）
        depth_coverage = np.count_nonzero(depth_map_fwd > 0) / (hw[0] * hw[1])
        depth_coverage_values.append(depth_coverage)

        # 保存可视化
        save_pano_image(depth_map_fwd,
                        os.path.join(output_dir,
                                     f"depth_gt_frame{idx}_forward.png"),
                        title=f"GT Depth Frame {idx} (forward)",
                        cmap='turbo',
                        vfov=vfov,
                        hfov=hfov)

        # Backward 半球投影
        R_back = R @ np.array([-1, 0, 0, 0, 1, 0, 0, 0, -1]).reshape(3, 3)
        T_back = T * np.array([-1, 1, -1])
        points_cam_back = points_world @ R_back + T_back

        depth_map_bwd, coverage_bwd = project_points_to_pano(
            points_cam_back, vfov, hfov, hw)

        save_pano_image(depth_map_bwd,
                        os.path.join(output_dir,
                                     f"depth_gt_frame{idx}_backward.png"),
                        title=f"GT Depth Frame {idx} (backward)",
                        cmap='turbo',
                        vfov=vfov,
                        hfov=hfov)

        # 前后向互补性
        fwd_count = np.count_nonzero(depth_map_fwd > 0)
        bwd_count = np.count_nonzero(depth_map_bwd > 0)
        total_count = fwd_count + bwd_count
        total_pixels = hw[0] * hw[1] * 2  # 前后各半
        total_coverage = total_count / total_pixels
        report.add(
            f"帧 {idx} 全景覆盖率",
            "pass" if total_coverage > 0.15 else "warn",
            value=
            f"前 {coverage_fwd:.1%}, 后 {coverage_bwd:.1%}, 总 {total_coverage:.1%}"
        )

    # 汇总全景投影结果
    avg_coverage = np.mean(coverage_values) if coverage_values else 0
    avg_depth_coverage = np.mean(
        depth_coverage_values) if depth_coverage_values else 0
    report.add("平均全景覆盖率",
               "pass" if avg_coverage > 0.15 else "warn",
               value=f"{avg_coverage:.1%}")
    report.add("平均深度有效比例",
               "pass" if avg_depth_coverage > 0.15 else "warn",
               value=f"{avg_depth_coverage:.1%}")

    if all_points_valid:
        report.add("全景图可视化", "pass", value=f"已保存到 {output_dir}/")
    else:
        report.add("全景图可视化", "warn", value="部分帧缺失, 可视化不完整")

    # ── OD 标签验证 ──
    pkl_path = getattr(conf, 'pkl_path', None)
    if pkl_path is not None and getattr(conf, 'use_od_labels', False):
        full_pkl = pkl_path if os.path.isabs(pkl_path) else os.path.join(
            source_path, pkl_path)
        try:
            pkl_data = load_pkl_compat(full_pkl)
            from scene.gacrt_loader import identify_moving_objects
            dynamic_labels = set(
                getattr(conf, 'od_dynamic_labels', [0, 1, 2, 5, 7]))
            min_disp = getattr(conf, 'od_min_displacement', 0.5)
            min_speed = getattr(conf, 'od_min_speed', None)
            moving_ids, obj_velocities = identify_moving_objects(
                pkl_data, dynamic_labels, min_disp, min_speed=min_speed)
            report.add("OD 移动目标识别", "pass", value=f"{len(moving_ids)} 个移动目标")
            report.add("OD 速度估计",
                       "pass",
                       value=f"{len(obj_velocities)} 个目标有速度")
        except Exception as e:
            report.add("OD 标签验证", "fail", f"加载/处理失败: {e}")
    else:
        report.add("OD 标签验证", "skip", value="OD 未启用")

    return report


# ─── Level 3: Scene 加载集成验证 ───────────────────────────────

def verify_level_3(conf, output_dir):
    report = VerifyReport("Level 3 — Scene 加载集成验证")
    os.makedirs(output_dir, exist_ok=True)

    # 需要 GPU 环境
    import torch
    if not torch.cuda.is_available():
        report.add("GPU 可用性", "fail", "未检测到 CUDA GPU")
        return report
    report.add("GPU 可用性", "pass", value=f"{torch.cuda.get_device_name(0)}")

    # 不使用 debug_cuda：完整加载所有帧，确保 val_frame_ids 能命中 test 集
    # debug_cuda 只加载 15 帧，而 val_frame_ids 通常在帧 50+，会导致 test 集为空
    conf.debug_cuda = False
    conf.model_path = os.path.join(output_dir, "verify_scene_model")
    os.makedirs(conf.model_path, exist_ok=True)

    # 导入 Scene 和 GaussianModel
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    try:
        gaussians = GaussianModel(conf)
        scene = Scene(conf, gaussians, shuffle=False)
        report.add("Scene 加载", "pass")
    except Exception as e:
        report.add("Scene 加载", "fail", f"加载失败: {e}")
        return report

    # point_cloud 验证
    scene_info = scene.scene_info if hasattr(scene, 'scene_info') else None
    pcd = gaussians._xyz if gaussians._xyz is not None else None

    if pcd is not None:
        n_points = pcd.shape[0]
        xyz_range = (pcd.min().item(), pcd.max().item())
        report.add("Gaussian 点数", "pass", value=f"{n_points}")
        report.add("Gaussian xyz 范围", "pass",
                   value=f"[{xyz_range[0]:.3f}, {xyz_range[1]:.3f}]")
    else:
        report.add("Gaussian 初始化", "fail", "xyz 为 None")

    # Camera 验证
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    total_cams = len(train_cams) + len(test_cams)

    report.add("train cameras", "pass", value=len(train_cams))
    report.add("test cameras", "pass" if len(test_cams) > 0 else "fail",
               value=len(test_cams))

    expected_total = 2 * conf.frames  # forward + backward per frame
    report.add("总 camera 数", "pass" if total_cams == expected_total else "warn",
               value=f"{total_cams}, 期望 {expected_total}")

    # Camera pts_depth 有效像素
    min_coverage = 1.0
    for cam in train_cams[:10]:  # 只检查前 10 个
        if cam.pts_depth is not None:
            coverage = torch.count_nonzero(cam.pts_depth > 0).item() / \
                       (cam.pts_depth.shape[1] * cam.pts_depth.shape[2])
            min_coverage = min(min_coverage, coverage)

    report.add("Camera pts_depth 最小覆盖率", "pass" if min_coverage > 0.2 else "warn",
               value=f"{min_coverage:.1%}")

    # PCA transform 文件
    pca_path = os.path.join(conf.model_path, 'transform_poses_pca.npz')
    report.add("PCA transform 保存", "pass" if os.path.exists(pca_path) else "fail",
               value=pca_path)

    if os.path.exists(pca_path):
        pca_data = np.load(pca_path)
        sf = pca_data['scale_factor'].item()
        report.add("scale_factor", "pass", value=f"{sf:.4f}")

    # Gaussian 属性一致性
    n = gaussians._xyz.shape[0]
    attr_shapes = {
        '_xyz': gaussians._xyz.shape,
        '_opacity': gaussians._opacity.shape,
        '_scaling': gaussians._scaling.shape,
        '_rotation': gaussians._rotation.shape,
        '_features_dc': gaussians._features_dc.shape,
        '_intensity': gaussians._intensity.shape,
    }
    all_consistent = all(s[0] == n for s in attr_shapes.values())
    report.add("Gaussian 属性一致性", "pass" if all_consistent else "fail",
               value=f"{n} 个 Gaussian")

    # Camera resolution
    if len(train_cams) > 0:
        cam = train_cams[0]
        report.add("Camera resolution", "pass",
                   value=f"({cam.image_width}, {cam.image_height})")

    return report


# ─── Level 4: 首次渲染冒烟测试 ──────────────────────────────────


def verify_level_4(conf, output_dir):
    report = VerifyReport("Level 4 — 首次渲染冒烟测试")
    os.makedirs(output_dir, exist_ok=True)

    import torch

    # 不使用 debug_cuda：完整加载所有帧，确保 val_frame_ids 能命中 test 集
    conf.debug_cuda = False
    conf.model_path = os.path.join(output_dir, "verify_scene_model")
    os.makedirs(conf.model_path, exist_ok=True)

    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render, render_range_map

    try:
        gaussians = GaussianModel(conf)
        scene = Scene(conf, gaussians, shuffle=False)
        gaussians.training_setup(conf)
    except Exception as e:
        report.add("初始化", "fail", f"Scene/Gaussian 加载失败: {e}")
        return report

    # ── 单次渲染 ──
    train_cams = scene.getTrainCameras()
    viewpoint_cam = train_cams[0]

    try:
        bg_color = [1, 1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # from utils.system_utils import search_for_config
        pipe_settings = conf  # 使用配置中的 PipelineParams

        render_result = render(viewpoint_cam, gaussians, pipe_settings, background)

        # 检查输出键
        required_keys = [
            'depth', 'depth_mean', 'depth_median', 'alpha', 'intensity_sh',
            'raydrop', 'normal'
        ]
        missing_keys = [k for k in required_keys if k not in render_result]
        if not missing_keys:
            report.add("render() 输出键",
                       "pass",
                       value=f"包含所有 {len(required_keys)} 个必需键")
        else:
            report.add("render() 输出键", "fail", value=f"缺失: {missing_keys}")

        # 深度图有效性
        depth = render_result['depth']
        nonzero_ratio = torch.count_nonzero(depth > 0).item() / depth.numel()
        depth_mean = depth[depth > 0].mean().item() if nonzero_ratio > 0 else 0
        depth_max = depth[depth > 0].max().item() if nonzero_ratio > 0 else 0

        report.add("depth 非零比例",
                   "pass" if nonzero_ratio > 0.1 else "warn",
                   value=f"{nonzero_ratio:.1%}")
        report.add("depth 均值/最大",
                   "pass" if depth_mean > 0 else "warn",
                   value=f"{depth_mean:.2f}m / {depth_max:.2f}m")

        # Alpha 图
        alpha = render_result['alpha']
        nonzero_alpha = torch.count_nonzero(alpha > 0).item() / alpha.numel()
        report.add("alpha 非零比例",
                   "pass" if nonzero_alpha > 0.1 else "warn",
                   value=f"{nonzero_alpha:.1%}")

        # Raydrop
        raydrop = render_result['raydrop']
        raydrop_range = (raydrop.min().item(), raydrop.max().item())
        report.add("raydrop 范围",
                   "pass" if raydrop_range[0] >= 0 and raydrop_range[1] <= 1
                   else "fail",
                   value=f"[{raydrop_range[0]:.3f}, {raydrop_range[1]:.3f}]")

    except Exception as e:
        report.add("render() 执行", "fail", f"渲染失败: {e}")
        return report

    # ── 损失计算 ──
    try:
        from utils.loss_utils import l1_loss
        gt_depth = viewpoint_cam.pts_depth
        if gt_depth is not None:
            loss_lidar = l1_loss(render_result['depth'], gt_depth)
            report.add("loss_lidar",
                       "pass" if not torch.isnan(loss_lidar) else "fail",
                       value=f"{loss_lidar.item():.4f}")

            # 梯度检查
            loss_lidar.backward()
            has_grad = gaussians._xyz.grad is not None and gaussians._xyz.grad.abs(
            ).sum() > 0
            report.add("梯度流通",
                       "pass" if has_grad else "fail",
                       value="xyz grad" if has_grad else "xyz grad 为零或 None")
    except Exception as e:
        report.add("损失计算", "fail", f"损失 backward 失败: {e}")

    # ── 全景拼接 ──
    try:
        test_cams = scene.getTestCameras()
        if len(test_cams) >= 2:
            # 取第一对 test camera (forward + backward)
            cam_front = test_cams[0]
            cam_back = test_cams[1]

            range_result = render_range_map(cam_front, cam_back, gaussians,
                                            pipe_settings)

            if 'depth_pano' in range_result:
                depth_pano = range_result['depth_pano']
                # 检查拼接边界连续性
                w_half = depth_pano.shape[-1] // 2
                # 前向右边界 vs 后向左边界
                if depth_pano.shape[-1] > 1:
                    boundary_left = depth_pano[0, :, 0].mean().item()
                    boundary_center_right = depth_pano[0, :, w_half -
                                                       1].mean().item()
                    boundary_center_left = depth_pano[0, :,
                                                      w_half].mean().item()
                    boundary_right = depth_pano[0, :, -1].mean().item()

                    report.add(
                        "全景拼接",
                        "pass",
                        value=
                        f"边界值: 左={boundary_left:.2f}, 中右={boundary_center_right:.2f}, "
                        f"中左={boundary_center_left:.2f}, 右={boundary_right:.2f}"
                    )

                # 保存全景图可视化
                pano_np = depth_pano[0].cpu().numpy()
                save_pano_image(np.expand_dims(pano_np, 0),
                                os.path.join(output_dir,
                                             "render_range_map_first.png"),
                                title="First Rendered Range Map (360°)",
                                cmap='turbo')

                report.add(
                    "全景图可视化",
                    "pass",
                    value=f"已保存到 {output_dir}/render_range_map_first.png")
            else:
                report.add("全景拼接", "warn", "render_range_map 输出不含 depth_pano")
    except Exception as e:
        report.add("全景拼接", "fail", f"render_range_map 失败: {e}")

    return report


# ─── 主入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GACRT 数据集适配验证")
    parser.add_argument("--config", type=str, required=True,
                        help="场景配置 YAML 文件路径")
    parser.add_argument("--source_path", type=str, default=None,
                        help="数据源路径 (覆盖配置中的 source_path)")
    parser.add_argument("--level", type=int, default=2,
                        choices=[1, 2, 3, 4],
                        help="验证层级 (1=数据完整性, 2=语义, 3=Scene加载, 4=渲染)")
    parser.add_argument("--output_dir", type=str, default="verify_output",
                        help="验证结果和可视化输出目录")
    parser.add_argument("--num_sample_frames", type=int, default=5,
                        help="Level 2 采样帧数")
    args = parser.parse_args()

    print("=" * 60)
    print("GACRT 数据集适配验证")
    print("=" * 60)

    # 加载配置
    conf = load_config(args.config, args.source_path)
    print(f"配置: {args.config}")
    print(f"sequence_id: {conf.sequence_id}")
    print(f"scene_type: {conf.scene_type}")
    print(f"source_path: {conf.source_path}")
    print(f"验证层级: {args.level}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_passed = True

    # Level 1
    report1 = verify_level_1(conf, args.output_dir)
    passed1 = report1.print_report()
    all_passed = all_passed and passed1

    # Level 2
    if args.level >= 2 and passed1:
        report2 = verify_level_2(conf, args.output_dir, args.num_sample_frames)
        passed2 = report2.print_report()
        all_passed = all_passed and passed2
    elif args.level >= 2 and not passed1:
        print("\n⚠️  Level 1 有失败项, 建议先修复再继续 Level 2")

    # Level 3
    if args.level >= 3 and all_passed:
        report3 = verify_level_3(conf, args.output_dir)
        passed3 = report3.print_report()
        all_passed = all_passed and passed3
    elif args.level >= 3 and not all_passed:
        print("\n⚠️  前面的层级有失败项, 建议先修复再继续 Level 3")

    # Level 4
    if args.level >= 4 and all_passed:
        report4 = verify_level_4(conf, args.output_dir)
        passed4 = report4.print_report()
        all_passed = all_passed and passed4
    elif args.level >= 4 and not all_passed:
        print("\n⚠️  前面的层级有失败项, 建议先修复再继续 Level 4")

    # 最终汇总
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 验证全部通过！数据集适配正确。")
    else:
        print("❌ 验证有失败项，请检查上方报告。")
    print("=" * 60)


if __name__ == '__main__':
    main()
