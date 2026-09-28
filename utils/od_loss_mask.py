#
# OD 3D bbox → 全景 range-image loss mask utilities.
# mask=1: pixel participates in static-background loss; mask=0: OD region (excluded).
#
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def scale_bbox(bbox, scale: float):
    return {
        "center": np.asarray(bbox["center"], dtype=np.float64),
        "size": np.asarray(bbox["size"], dtype=np.float64) * scale,
        "yaw": float(bbox["yaw"]),
    }


def transform_points_pca(points_world, transform: np.ndarray):
    """Same homogeneous transform used for the LiDAR point cloud.

    `transform` already includes the scene scale_factor on its linear part.
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (pts_h @ transform.T)[:, :3]


def transform_bboxes_pca(bboxes, transform: np.ndarray, scale_factor: float):
    """Apply the same PCA + scale used for points: p' = (p_h @ transform.T)[:3].

    `transform` already contains scale_factor. Do not scale the center again.
    Box size must be multiplied by scale_factor; yaw follows the PCA rotation.
    """
    scale = float(scale_factor) if scale_factor else 1.0
    rot = transform[:3, :3] / max(abs(scale), 1e-12)
    out = []
    for bbox in bboxes:
        center_h = np.array([*bbox["center"], 1.0], dtype=np.float64)
        center_pca = (center_h @ transform.T)[:3]
        heading = np.array([np.cos(bbox["yaw"]), np.sin(bbox["yaw"]), 0.0], dtype=np.float64)
        heading_pca = heading @ rot.T
        out.append({
            "center": center_pca,
            "size": np.asarray(bbox["size"], dtype=np.float64) * scale,
            "yaw": float(np.arctan2(heading_pca[1], heading_pca[0])),
        })
    return out


def sample_points_in_bbox(bbox, n_per_axis: int = 8):
    """Uniform grid samples inside an oriented 3D bbox (world frame)."""
    center = np.asarray(bbox["center"], dtype=np.float64)
    half = np.asarray(bbox["size"], dtype=np.float64) * 0.5
    yaw = float(bbox["yaw"])
    lin = np.linspace(-1.0, 1.0, n_per_axis)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    local = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) * half
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    world = local @ rot.T + center
    return world


def sample_bbox_surface(bbox, n_face: int = 16):
    """Dense samples on the 6 faces of an oriented 3D bbox (world frame)."""
    center = np.asarray(bbox["center"], dtype=np.float64)
    half = np.asarray(bbox["size"], dtype=np.float64) * 0.5
    yaw = float(bbox["yaw"])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    lin = np.linspace(-1.0, 1.0, n_face)
    a, b = np.meshgrid(lin, lin, indexing="ij")
    ones = np.ones_like(a)
    faces = [
        np.stack([ones, a, b], axis=-1).reshape(-1, 3),
        np.stack([-ones, a, b], axis=-1).reshape(-1, 3),
        np.stack([a, ones, b], axis=-1).reshape(-1, 3),
        np.stack([a, -ones, b], axis=-1).reshape(-1, 3),
        np.stack([a, b, ones], axis=-1).reshape(-1, 3),
        np.stack([a, b, -ones], axis=-1).reshape(-1, 3),
    ]
    local = np.concatenate(faces, axis=0) * half
    return local @ rot.T + center


def project_cam_to_pano_pixels(points_cam, h: int, w: int, vfov, hfov):
    """Project camera-frame points to integer panorama pixels (matches camera_utils.loadCam)."""
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    phi = np.arctan2(x, z)
    theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)

    vfov_max = np.pi / 2 - vfov[0] * np.pi / 180
    vfov_min = np.pi / 2 - vfov[1] * np.pi / 180
    hfov_max = hfov[1] * np.pi / 180
    hfov_min = hfov[0] * np.pi / 180

    row = (theta - vfov_min) * h / (vfov_max - vfov_min)
    col = (phi - hfov_min) * w / (hfov_max - hfov_min)
    uv = np.stack([row, col], axis=1)
    uv = uv[(uv[:, 0] >= -0.5) & (uv[:, 0] < h - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < w - 0.5)]
    return np.round(uv).astype(np.int32)


def _fill_convex_silhouette(mask_hw, uv_rc, h, w):
    """Write 0 into mask_hw (H, W) for the convex hull of uv_rc (row, col)."""
    if uv_rc.shape[0] == 0:
        return
    if uv_rc.shape[0] < 3:
        mask_hw[uv_rc[:, 0], uv_rc[:, 1]] = 0.0
        return
    pts_xy = np.stack([uv_rc[:, 1], uv_rc[:, 0]], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts_xy)
    canvas = np.ones((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, hull, 0)
    mask_hw *= canvas.astype(np.float32)


def fill_pano_silhouette(mask_hw, uv, h, w):
    """Fill the projected silhouette, splitting left/right if the bbox wraps the pano seam."""
    if uv.shape[0] == 0:
        return
    col = uv[:, 1]
    if col.max() - col.min() > 0.55 * w:
        _fill_convex_silhouette(mask_hw, uv[col < w * 0.5], h, w)
        _fill_convex_silhouette(mask_hw, uv[col >= w * 0.5], h, w)
    else:
        _fill_convex_silhouette(mask_hw, uv, h, w)


def build_od_loss_mask(R, T, bboxes, h: int, w: int, vfov, hfov, bbox_scale: float = 1.15):
    """Return (1, H, W) float32 mask: 1=valid for loss, 0=filled OD bbox silhouette."""
    mask = np.ones((1, h, w), dtype=np.float32)
    if not bboxes:
        return mask

    mask_hw = mask[0]
    for bbox in bboxes:
        scaled = scale_bbox(bbox, bbox_scale)
        pts_world = sample_bbox_surface(scaled)
        pts_cam = pts_world @ R + T
        uv = project_cam_to_pano_pixels(pts_cam, h, w, vfov, hfov)
        fill_pano_silhouette(mask_hw, uv, h, w)

    return mask


def resize_od_mask(od_mask, h, w):
    """Resize OD mask to (1, h, w). Accepts numpy or torch."""
    if od_mask is None:
        return None
    if not torch.is_tensor(od_mask):
        od_mask = torch.from_numpy(od_mask).float()
    if od_mask.ndim == 2:
        od_mask = od_mask.unsqueeze(0)
    if od_mask.shape[-2:] != (h, w):
        od_mask = F.interpolate(od_mask.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return od_mask


def stitch_od_mask_pano(cam_front, cam_back, hw):
    """OD mask: HxW if ``cam_back is None``, else 360 pano (1, H, 2W). 1=valid, 0=OD."""
    h, w = hw
    if cam_back is None:
        m = getattr(cam_front, "od_loss_mask", None)
        if m is None:
            return torch.ones((1, h, w), dtype=torch.float32, device="cuda")
        return resize_od_mask(m, h, w).to(device="cuda", dtype=torch.float32)

    breaks = (0, w // 2, 3 * w // 2, w * 2)
    od_pano = torch.ones((1, h, w * 2), dtype=torch.float32, device="cuda")

    for idx, viewpoint in enumerate([cam_front, cam_back]):
        m = getattr(viewpoint, "od_loss_mask", None)
        if m is None:
            continue
        m = resize_od_mask(m, h, w).to(device="cuda", dtype=torch.float32)
        if idx == 0:
            od_pano[:, :, breaks[1]:breaks[2]] = m
        else:
            od_pano[:, :, breaks[2]:breaks[3]] = m[:, :, 0:(breaks[3] - breaks[2])]
            od_pano[:, :, breaks[0]:breaks[1]] = m[:, :, (w - breaks[1] + breaks[0]):w]
    return od_pano
