# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GS-LiDAR is the official implementation of "GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting" (ICLR 2025, Fudan Zhang Vision Group). It generates realistic LiDAR point clouds by applying panoramic Gaussian splatting to LiDAR data, using 2D Gaussian splatting with a custom panoramic rasterizer and a RayDrop prior (UNet) to model LiDAR ray dropout patterns.

## Environment Setup

```bash
conda create --name gslidar python=3.9
conda activate gslidar
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2  # adjust for your CUDA version
pip install -r requirements.txt

# External dependencies requiring compilation
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

cd chamfer/chamfer3D && python setup.py install && cd ../..

cd diff-gaussian-rasterization-2d && pip install . && cd ..
```

## Key Commands

### Data Preprocessing
```bash
python preprocess/kitti360_to_gs.py --seq {sequence_id}
# Supported sequence IDs: 1538, 1728, 1908, 3353, 2350, 4950, 8120, 10200, 10750, 11400
```

### Training
```bash
# Static scene
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/kitti360_nvs_1908.yaml \
  source_path=data/kitti360 \
  model_path=eval_output/kitti360_reconstruction/1908

# Dynamic scene
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/kitti360_nvs_10750.yaml \
  source_path=data/kitti360 \
  model_path=eval_output/kitti360_reconstruction/10750
```

### Evaluation Only
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/kitti360_nvs_1908.yaml \
  source_path=data/kitti360 \
  model_path=eval_output/kitti360_reconstruction/1908 \
  --test_only
```

Additional flags: `--show_log` (print to terminal instead of log.txt), `--median_depth`, `--debug_cuda` (use lowest resolution).

### Visualization
```bash
python scripts/visualize_lidar_in_video.py --seq {sequence_id}
python scripts/compare_2dgs_3dgs.py  # educational: 2DGS vs 3DGS comparison
```

## Architecture

### Pipeline: Training → Refine → Refine-Test

`train.py` runs three stages sequentially:
1. **`training()`** — Optimizes GaussianModel + RayDropPrior via multi-loss training (depth, intensity, raydrop, chamfer, etc.). Saves checkpoints and renders evaluation at `test_iterations`.
2. **`refine()`** — Trains a UNet on the rendered raydrop datasets to refine raydrop prediction. 1000 epochs of BCE loss with random masking augmentation.
3. **`refine_test()`** — Evaluates the refined UNet on both train and test splits.

### Core Modules

- **`scene/gaussian_model.py`** — `GaussianModel` class: stores and optimizes per-Gaussian attributes (xyz, scaling, rotation, opacity, SH features, time `t`, temporal scaling `scaling_t`, velocity, intensity). Handles densification/pruning logic. Key distinction from standard 3DGS: adds temporal dimensions (`_t`, `_scaling_t`, `_velocity`) for dynamic scenes.

- **`scene/__init__.py`** — `Scene` class: loads dataset via scene-type callbacks (currently only `"Kitti360"`), manages train/test cameras at multiple resolution scales, and initializes Gaussians from point cloud data.

- **`gaussian_renderer/__init__.py`** — `render()` and `render_range_map()`: panoramic Gaussian splatting using a custom 2D rasterizer (`diff_gaussian_rasterization_2d`). The rasterizer takes `vfov`/`hfov` for panoramic projection. When `neg_fov=true`, uses `tan(-0.5)` for both tanfov values (empirically better results).

- **`scene/raydrop_prior.py`** — `RayDropPrior`: a learnable environment map that predicts per-pixel raydrop probability, upscalable as training resolution increases.

- **`scene/unet.py`** — UNet used in the refine stage for post-hoc raydrop refinement.

### Data Flow

1. `preprocess/kitti360_to_gs.py` → generates `transforms_{seq}_all.json` with lidar2world poses
2. `scene/kitti360_loader.py` → reads JSON + KITTI-360 raw data → creates `Camera` objects with panoramic depth/intensity maps
3. Training loop: random viewpoint sampling → `render()` → multi-loss backward → densify/prune → repeat
4. Evaluation: `render_range_map()` stitches front/back panoramas, applies raydrop mask, computes metrics (Chamfer distance, F-score, depth RMSE/LPIPS/SSIM/PSNR, intensity, raydrop F1)

### Configuration System

Uses OmegaConf with layered merge: `configs/base.yaml` ← scene-specific YAML ← CLI overrides. The base config defines all parameters with defaults; scene configs override only what's needed (e.g., `sequence_id`, `dynamic`, loss weights, `vfov`/`hfov`). CLI args like `source_path` and `model_path` are passed as `key=value` after `--config`.

### Custom CUDA Extensions

- **`diff-gaussian-rasterization-2d/`** — Forked/modified 2D Gaussian splatting rasterizer with panoramic FoV support (`vfov`, `hfov` params). Must be compiled with matching CUDA toolkit.
- **`chamfer/chamfer3D/`** — Chamfer distance CUDA extension for point cloud metric computation.

## Dataset Layout

```
data/kitti360/
├── KITTI-360/
│   ├── calibration/
│   ├── data_3d_raw/
│   └── data_poses/
└── {sequence_id}/
    └── transforms_{sequence_id}_all.json
```
