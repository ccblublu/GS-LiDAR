#
# Static observation confidence map for hybrid hole-filling policy.
#
import os

import numpy as np
import torch


class StaticConfidenceTracker:
    """Accumulate per-pixel static GT supervision counts during training."""

    def __init__(self, h: int, w: int, pano_width: int = None):
        self.h = h
        self.w = w if pano_width is None else pano_width
        self.counts = np.zeros((self.h, self.w), dtype=np.int32)

    def update(self, pts_depth, od_loss_mask=None):
        """Record pixels that received valid static supervision."""
        if pts_depth is None:
            return
        depth = pts_depth.detach().cpu().numpy() if torch.is_tensor(pts_depth) else pts_depth
        valid = depth[0] > 0 if depth.ndim == 3 else depth > 0
        if od_loss_mask is not None:
            om = od_loss_mask.detach().cpu().numpy() if torch.is_tensor(od_loss_mask) else od_loss_mask
            if om.ndim == 3:
                om = om[0]
            valid = valid & (om > 0.5)
        self.counts[valid] += 1

    def update_from_camera(self, viewpoint_cam):
        if not hasattr(viewpoint_cam, "od_loss_mask") or viewpoint_cam.od_loss_mask is None:
            mask = None
        else:
            mask = viewpoint_cam.od_loss_mask
        self.update(viewpoint_cam.pts_depth, mask)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, self.counts)

    @classmethod
    def load(cls, path: str):
        counts = np.load(path)
        tracker = cls(counts.shape[0], counts.shape[1])
        tracker.counts = counts.astype(np.int32)
        return tracker

    def query_mask(self, threshold: int = 1):
        """High-confidence pixels observed at least `threshold` times as static."""
        return self.counts >= threshold

    def apply_hole_policy(self, depth_pano, intensity_pano, hole_mask, policy: str = "raydrop"):
        """Apply hole policy on composited pano where depth is missing."""
        depth = depth_pano.clone() if torch.is_tensor(depth_pano) else torch.from_numpy(depth_pano).clone()
        intensity = intensity_pano.clone() if torch.is_tensor(intensity_pano) else torch.from_numpy(intensity_pano).clone()

        if not torch.is_tensor(hole_mask):
            hole_mask = torch.from_numpy(hole_mask)

        low_conf = hole_mask & (~torch.from_numpy(self.query_mask()).to(hole_mask.device))

        if policy == "raydrop":
            depth = depth * (~low_conf).float()
            intensity = intensity * (~low_conf).float()
        elif policy == "keep":
            pass
        else:
            raise ValueError(f"Unknown hole policy: {policy}")

        return depth, intensity
