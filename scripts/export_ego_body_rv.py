#!/usr/bin/env python3
"""Export per-frame GT AABB body RV packs for refine_test re-paste.

Writes ``{model_path}/ray_drop_datasets/ego_body/{colmap_id:03d}.pt``
as [body_depth, body_inten, clear_mask] without re-rendering Gaussians.

Example:
  python scripts/export_ego_body_rv.py \\
    --config configs/gacrt_nvs_PCI_20260825_qt128.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.ego_body_mask import build_gt_ego_body_rv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    args_cli, unknown = parser.parse_known_args()
    conf = OmegaConf.merge(
        OmegaConf.load(args_cli.base_config),
        OmegaConf.load(args_cli.config),
        OmegaConf.from_dotlist([u for u in unknown if "=" in u]),
    )

    # Lowest-res scale is enough: we only need camera ego_body_points.
    if conf.resolution_scales:
        conf.resolution_scales = [conf.resolution_scales[-1]]

    gaussians = GaussianModel(conf.sh_degree)
    scene = Scene(conf, gaussians, shuffle=False)
    h, w = int(conf.hw[0]), int(conf.hw[1])
    # Use native hw for body RV (cameras may be at last scale — rebuild at full hw
    # via build_gt which resizes masks / projects at given hw).
    scale = conf.resolution_scales[0]
    h_s, w_s = h // scale, w // scale
    dilate = int(getattr(conf, "ego_body_replace_dilate", 2) or 0)
    out_dir = Path(conf.model_path) / "ray_drop_datasets" / "ego_body"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for split_name, getter in (("train", scene.getTrainCameras),
                               ("test", scene.getTestCameras)):
        cams = getter(scale=scale) if split_name == "test" else getter()
        if not cams:
            continue
        for i in tqdm(range(len(cams) // 2), desc=f"ego_body {split_name}"):
            cam_f, cam_b = cams[i * 2], cams[i * 2 + 1]
            body_d, body_i, clear = build_gt_ego_body_rv(
                cam_f, cam_b, [h_s, w_s], conf.vfov, conf.hfov,
                dilate=dilate, device="cpu")
            if body_d is None:
                continue
            path = out_dir / f"{cam_f.colmap_id:03d}.pt"
            torch.save(torch.cat([body_d, body_i, clear]), path)
            n_saved += 1

    print(f"[export] wrote {n_saved} packs → {out_dir}")


if __name__ == "__main__":
    main()
