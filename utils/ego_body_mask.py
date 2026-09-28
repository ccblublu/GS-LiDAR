#
# Ego-body handling: ignore in training loss; at inference composite a rigid
# ego-body point cloud (like an image ego mask paste), instead of leaving holes
# or unsupervised floaters.
#
import os
import numpy as np
import torch
import torch.nn.functional as F

from utils.od_loss_mask import resize_od_mask


def resolve_ego_body_cloud_path(args):
    """Config path, else ``{model_path}/ego_body_imu.npz`` if present."""
    explicit = getattr(args, "ego_body_cloud", None)
    if explicit:
        return str(explicit)
    model_path = getattr(args, "model_path", None)
    if not model_path:
        return None
    default = os.path.join(model_path, "ego_body_imu.npz")
    return default if os.path.isfile(default) else None


def get_camera_ego_body_mask(cam, h, w, device=None):
    """Return (1, H, W) float mask, 1 = body pixel. None if unavailable."""
    ego = getattr(cam, "ego_body_mask", None)
    if ego is None:
        return None
    ego = resize_od_mask(ego, h, w)
    if device is not None:
        ego = ego.to(device)
    return ego


def stitch_ego_body_mask(cam_front, cam_back, hw):
    """Ego body mask: HxW if ``cam_back is None``, else 360 pano (1, H, 2W). 1 = body."""
    h, w = hw
    front = get_camera_ego_body_mask(cam_front, h, w, device="cuda")
    if cam_back is None:
        return front
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    back = get_camera_ego_body_mask(cam_back, h, w, device="cuda")
    if front is None or back is None:
        return None
    out = torch.zeros((1, h, w * 2), dtype=torch.float32, device=front.device)
    out[:, :, breaks[1]:breaks[2]] = front
    out[:, :, breaks[2]:breaks[3]] = back[:, :, 0:(breaks[3] - breaks[2])]
    out[:, :, breaks[0]:breaks[1]] = back[:, :, (w - breaks[1] + breaks[0]):w]
    return out


def points_in_aabb_np(points, aabb):
    box = np.asarray(aabb, dtype=np.float64)
    if box.shape != (3, 2) or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    return (
        (points[:, 0] >= box[0, 0]) & (points[:, 0] <= box[0, 1]) &
        (points[:, 1] >= box[1, 0]) & (points[:, 1] <= box[1, 1]) &
        (points[:, 2] >= box[2, 0]) & (points[:, 2] <= box[2, 1])
    )


def filter_points_outside_aabb(points, aabb, intensity=None):
    """Drop points that fall inside the ego body AABB (numpy, IMU/ego frame)."""
    if aabb is None or points is None or len(points) == 0:
        return points, intensity
    inside = points_in_aabb_np(np.asarray(points), aabb)
    keep = ~inside
    pts = np.asarray(points)[keep]
    if intensity is None:
        return pts, None
    return pts, np.asarray(intensity)[keep]


def save_ego_body_cloud(path, xyz_imu, intensity=None, max_points=200000):
    """Persist a rigid ego-body template in the IMU / ego frame."""
    xyz = np.asarray(xyz_imu, dtype=np.float32)
    if intensity is None:
        intensity = np.zeros((len(xyz),), dtype=np.float32)
    else:
        intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if len(intensity) and intensity.max() > 1.5:
        intensity = intensity / 255.0
    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), max_points, replace=False)
        xyz, intensity = xyz[sel], intensity[sel]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, xyz=xyz, intensity=intensity)
    return path


def load_ego_body_cloud(path):
    if path is None or not os.path.isfile(path):
        return None, None
    data = np.load(path)
    xyz = data["xyz"].astype(np.float32)
    inten = data["intensity"].astype(np.float32)
    if len(inten) and inten.max() > 1.5:
        inten = inten / 255.0
    return xyz, inten


def _project_body_half(points_cam, intensity, h, w, vfov, hfov):
    from utils.camera_utils import project_points_to_rv
    if points_cam is None or len(points_cam) == 0:
        return (np.zeros((1, h, w), dtype=np.float32),
                np.zeros((1, h, w), dtype=np.float32))
    inten = intensity
    if inten is None:
        inten = np.ones((len(points_cam),), dtype=np.float32)
    return project_points_to_rv(points_cam, inten, h, w, vfov, hfov)


def stitch_projected_body(cam_front, cam_back, hw, vfov, hfov):
    """Project ego body points: HxW if ``cam_back is None``, else 360 Hx(2W)."""
    h, w = hw
    d_f, i_f = _project_body_half(
        getattr(cam_front, "ego_body_points_cam", None),
        getattr(cam_front, "ego_body_intensity", None),
        h, w, vfov, hfov)
    if cam_back is None:
        return d_f, i_f
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    d_b, i_b = _project_body_half(
        getattr(cam_back, "ego_body_points_cam", None),
        getattr(cam_back, "ego_body_intensity", None),
        h, w, vfov, hfov)
    depth = np.zeros((1, h, w * 2), dtype=np.float32)
    inten = np.zeros((1, h, w * 2), dtype=np.float32)
    depth[:, :, breaks[1]:breaks[2]] = d_f
    inten[:, :, breaks[1]:breaks[2]] = i_f
    depth[:, :, breaks[2]:breaks[3]] = d_b[:, :, 0:(breaks[3] - breaks[2])]
    depth[:, :, breaks[0]:breaks[1]] = d_b[:, :, (w - breaks[1] + breaks[0]):w]
    inten[:, :, breaks[2]:breaks[3]] = i_b[:, :, 0:(breaks[3] - breaks[2])]
    inten[:, :, breaks[0]:breaks[1]] = i_b[:, :, (w - breaks[1] + breaks[0]):w]
    return depth, inten


def _dilate_mask(mask_bool, k: int = 2):
    """Cheap binary dilate on (1,H,W) bool/float tensor."""
    if k <= 0:
        return mask_bool.float()
    x = mask_bool.float().unsqueeze(0)  # (1,1,H,W)
    x = F.max_pool2d(x, kernel_size=2 * k + 1, stride=1, padding=k)
    return x.squeeze(0)


def build_gt_ego_body_rv(cam_front, cam_back, hw, vfov, hfov, dilate: int = 2, device=None):
    """RV-project this frame's GT AABB body points.

    Returns HxW when ``cam_back is None``, else stitched Hx(2W).
    Each of body_depth / body_inten / clear_mask is (1, H, W|2W) float,
    or (None, None, None) if unavailable.
    """
    has_pts = getattr(cam_front, "ego_body_points_cam", None) is not None
    has_mask = getattr(cam_front, "ego_body_mask", None) is not None
    if cam_back is not None:
        has_pts = has_pts or getattr(cam_back, "ego_body_points_cam", None) is not None
        has_mask = has_mask or getattr(cam_back, "ego_body_mask", None) is not None
    if not has_pts and not has_mask:
        return None, None, None

    body_depth_np, body_inten_np = stitch_projected_body(
        cam_front, cam_back, hw, vfov, hfov)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    body_depth = torch.from_numpy(body_depth_np).to(device=device, dtype=torch.float32)
    body_inten = torch.from_numpy(body_inten_np).to(device=device, dtype=torch.float32)

    ego_mask = stitch_ego_body_mask(cam_front, cam_back, hw)
    if ego_mask is None:
        ego_mask = (body_depth > 0).float()
    else:
        ego_mask = ego_mask.to(device=device, dtype=torch.float32)
        if ego_mask.shape[-2:] != body_depth.shape[-2:]:
            ego_mask = F.interpolate(
                ego_mask.unsqueeze(0), size=body_depth.shape[-2:], mode="nearest").squeeze(0)
        # Clear wherever GT body projects, plus sparse ignore holes from loader.
        ego_mask = torch.maximum(ego_mask, (body_depth > 0).to(ego_mask.dtype))

    if dilate > 0:
        ego_mask = _dilate_mask(ego_mask > 0.5, dilate)
    clear_mask = (ego_mask > 0.5).float()
    return body_depth, body_inten, clear_mask


def paste_gt_ego_body_rv(depth, intensity, raydrop, body_depth, body_inten, clear_mask):
    """Replace inference RV in ``clear_mask`` with GT-projected body returns.

    Used at render-time and again after UNet raydrop (which can wipe the paste).
    """
    if depth is None or body_depth is None or clear_mask is None:
        return depth, intensity, raydrop

    device, dtype = depth.device, depth.dtype
    body_depth = body_depth.to(device=device, dtype=dtype)
    body_inten = body_inten.to(device=device, dtype=dtype) if body_inten is not None else None
    clear = clear_mask.to(device=device, dtype=dtype) > 0.5
    if clear.shape[-2:] != depth.shape[-2:]:
        clear = F.interpolate(
            clear.float().unsqueeze(0), size=depth.shape[-2:], mode="nearest"
        ).squeeze(0) > 0.5
        body_depth = F.interpolate(
            body_depth.unsqueeze(0), size=depth.shape[-2:], mode="nearest").squeeze(0)
        if body_inten is not None:
            body_inten = F.interpolate(
                body_inten.unsqueeze(0), size=depth.shape[-2:], mode="nearest").squeeze(0)

    body_hit = clear & (body_depth > 0)

    depth = depth.clone()
    for c in range(depth.shape[0]):
        depth[c] = torch.where(clear[0], torch.zeros_like(depth[c]), depth[c])
        depth[c] = torch.where(body_hit[0], body_depth[0], depth[c])

    if intensity is not None and body_inten is not None:
        intensity = intensity.clone()
        intensity[0] = torch.where(clear[0], torch.zeros_like(intensity[0]), intensity[0])
        intensity[0] = torch.where(body_hit[0], body_inten[0], intensity[0])

    if raydrop is not None:
        raydrop = raydrop.clone()
        # Cleared floater region → drop; GT body hits → keep.
        raydrop[0] = torch.where(clear[0], torch.ones_like(raydrop[0]), raydrop[0])
        raydrop[0] = torch.where(body_hit[0], torch.zeros_like(raydrop[0]), raydrop[0])

    return depth, intensity, raydrop


def composite_ego_body_pano(depth, intensity, raydrop, cam_front, cam_back, hw, vfov, hfov,
                            dilate: int = 2):
    """Inference: RV-project this frame's GT AABB body points and paste over render.

    Works for single-half (``cam_back is None``, HxW) or stitched 360 (Hx 2W).
    """
    if depth is None:
        return depth, intensity, raydrop
    body_depth, body_inten, clear_mask = build_gt_ego_body_rv(
        cam_front, cam_back, hw, vfov, hfov, dilate=dilate, device=depth.device)
    return paste_gt_ego_body_rv(
        depth, intensity, raydrop, body_depth, body_inten, clear_mask)


def append_ego_body_world_points(scene_xyz, scene_intensity, body_imu_xyz, body_imu_inten,
                                 T_imu_to_world):
    """Concatenate rigid ego-body points (IMU → world) onto a scene cloud."""
    if body_imu_xyz is None or len(body_imu_xyz) == 0:
        return scene_xyz, scene_intensity
    homo = np.concatenate(
        [np.asarray(body_imu_xyz, dtype=np.float64),
         np.ones((len(body_imu_xyz), 1), dtype=np.float64)], axis=1)
    body_w = (homo @ np.asarray(T_imu_to_world, dtype=np.float64).T)[:, :3].astype(np.float32)
    if body_imu_inten is None:
        body_imu_inten = np.zeros((len(body_w),), dtype=np.float32)
    body_i = np.asarray(body_imu_inten, dtype=np.float32).reshape(-1)
    if scene_xyz is None or len(scene_xyz) == 0:
        return body_w, body_i
    xyz = np.concatenate([np.asarray(scene_xyz), body_w], axis=0)
    if scene_intensity is None:
        return xyz, body_i
    inten = np.concatenate([np.asarray(scene_intensity).reshape(-1), body_i], axis=0)
    return xyz, inten
