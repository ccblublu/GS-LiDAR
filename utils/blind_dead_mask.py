#
# Static per-sensor dead zone for blind lidars (self-occlusion by the ego body /
# mount). A pixel that no training frame ever hits is a sensor property, not
# scene raydrop: excluded from raydrop loss, forced to drop at inference.
#
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CACHE_NAME = "blind_dead_mask.npz"


def dead_mask_enabled(args):
    return bool(getattr(args, "blind_dead_mask", False))


def _key(sensor_id, towards):
    return f"{int(sensor_id or 0)}_{towards or 'forward'}"


def _occupancy(points_cam, h, w, vfov, hfov):
    """Same pixel rounding as ``project_points_to_rv``, without the z-buffer."""
    occ = np.zeros((h, w), dtype=bool)
    if points_cam is None or len(points_cam) == 0:
        return occ
    p = np.asarray(points_cam, dtype=np.float64)
    phi = np.arctan2(p[:, 0], p[:, 2])
    theta = np.arctan2(np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2), -p[:, 1])
    v_max = np.pi / 2 - vfov[0] * np.pi / 180
    v_min = np.pi / 2 - vfov[1] * np.pi / 180
    h_max = hfov[1] * np.pi / 180
    h_min = hfov[0] * np.pi / 180
    row = np.around((theta - v_min) * h / (v_max - v_min)).astype(np.int64)
    col = np.around((phi - h_min) * w / (h_max - h_min)).astype(np.int64)
    keep = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    occ[row[keep], col[keep]] = True
    return occ


def _keep_bottom_connected(dead):
    """Keep dead components touching the bottom row: body occlusion, not open sky."""
    n, labels = cv2.connectedComponents(dead.astype(np.uint8), connectivity=8)
    if n <= 1:
        return dead
    bottom = np.unique(labels[-1][labels[-1] > 0])
    return np.isin(labels, bottom)


def compute_dead_masks(cam_infos, h, w, vfov, hfov, thresh=0.02, col_window=2,
                       bottom_connected=True):
    """Return {key: (H, W) bool dead} and {key: (H, W) float hit frequency}."""
    hits, counts = {}, {}
    for cam in cam_infos:
        k = _key(getattr(cam, "sensor_id", 0), getattr(cam, "towards", "forward"))
        occ = _occupancy(cam.pointcloud_camera, h, w, vfov, hfov)
        body = getattr(cam, "ego_body_points_cam", None)
        if body is not None and len(body):
            occ |= _occupancy(body, h, w, vfov, hfov)
        if k not in hits:
            hits[k] = np.zeros((h, w), dtype=np.float64)
            counts[k] = 0
        hits[k] += occ
        counts[k] += 1

    dead, freq = {}, {}
    for k in hits:
        f = hits[k] / max(counts[k], 1)
        # The RV grid is finer than QT128 azimuth sampling; bridge the empty columns.
        if col_window > 0:
            f = np.maximum.reduce([np.roll(f, s, axis=1) for s in range(-col_window, col_window + 1)])
        d = f < thresh
        if bottom_connected:
            d = _keep_bottom_connected(d)
        dead[k] = d
        freq[k] = f.astype(np.float32)
    return dead, freq


def ignore_from_dead(dead, dilate=2):
    """Dead zone plus a boundary band whose edge jitters with motion deskew."""
    if dilate <= 0:
        return dead.copy()
    kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
    return cv2.dilate(dead.astype(np.uint8), kernel) > 0


def save_dead_masks(path, dead, freq, hw):
    payload = {"hw": np.asarray(hw, dtype=np.int64)}
    for k in dead:
        payload[f"dead_{k}"] = dead[k]
        payload[f"freq_{k}"] = freq[k]
    np.savez_compressed(path, **payload)


def load_dead_masks(path):
    data = np.load(path)
    dead = {n[len("dead_"):]: data[n].astype(bool) for n in data.files if n.startswith("dead_")}
    return dead, tuple(int(x) for x in data["hw"])


def build_dead_masks(train_cam_infos, args):
    """Compute (or reuse the cached) dead masks. Returns ({key: dead}, {key: ignore})."""
    h, w = int(args.hw[0]), int(args.hw[1])
    cache = os.path.join(args.model_path, CACHE_NAME)
    reuse = (args.test_only or getattr(args, "start_checkpoint", None)) and os.path.isfile(cache)
    dead = None
    if reuse:
        dead, cached_hw = load_dead_masks(cache)
        if cached_hw != (h, w):
            print(f"[Blind dead] cache hw={cached_hw} != {(h, w)}, recomputing")
            dead = None
        else:
            print(f"[Blind dead] reused {cache}")
    if dead is None:
        dead, freq = compute_dead_masks(
            train_cam_infos, h, w, args.vfov, args.hfov,
            thresh=float(getattr(args, "blind_dead_thresh", 0.02)),
            col_window=int(getattr(args, "blind_dead_col_window", 2)),
            bottom_connected=bool(getattr(args, "blind_dead_bottom_connected", True)))
        os.makedirs(args.model_path, exist_ok=True)
        save_dead_masks(cache, dead, freq, (h, w))
        print(f"[Blind dead] saved {cache}")

    dilate = int(getattr(args, "blind_dead_ignore_dilate", 2))
    ignore = {k: ignore_from_dead(d, dilate) for k, d in dead.items()}
    for k in sorted(dead):
        print(f"[Blind dead] {k}: dead={dead[k].mean():.3f} ignore={ignore[k].mean():.3f} of RV")
    return dead, ignore


def attach_dead_masks(cam_infos, dead, ignore):
    """Set ``blind_dead_mask`` / ``blind_ignore_mask`` on each CameraInfo."""
    out = []
    for cam in cam_infos:
        k = _key(getattr(cam, "sensor_id", 0), getattr(cam, "towards", "forward"))
        if k not in dead:
            out.append(cam)
            continue
        out.append(cam._replace(
            blind_dead_mask=dead[k][None].astype(np.float32),
            blind_ignore_mask=ignore[k][None].astype(np.float32)))
    return out


def resize_dead_mask(mask, h, w, any_pixel):
    """Downsample (1, H, W): ``any_pixel`` → ignore semantics, else majority for dead."""
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.from_numpy(np.asarray(mask)).float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[-2:] == (h, w):
        return mask.float()
    area = F.interpolate(mask.float().unsqueeze(0), size=(h, w), mode="area").squeeze(0)
    return (area > 0).float() if any_pixel else (area >= 0.5).float()


_CACHE = {}


def _cached_for_args(args):
    """Fallback for cameras built outside the loader (novel-pose scripts)."""
    model_path = getattr(args, "model_path", None)
    if not model_path:
        return None
    path = os.path.join(model_path, CACHE_NAME)
    if path not in _CACHE:
        _CACHE[path] = load_dead_masks(path)[0] if os.path.isfile(path) else None
    return _CACHE[path]


def get_camera_mask(cam, h, w, args, attr="blind_dead_mask"):
    """(1, h, w) float mask on the camera, else from the model_path cache. None if absent."""
    if not dead_mask_enabled(args):
        return None
    m = getattr(cam, attr, None)
    if m is None:
        cached = _cached_for_args(args)
        if cached is None:
            return None
        d = cached.get(_key(getattr(cam, "sensor_id", 0), getattr(cam, "towards", "forward")))
        if d is None:
            return None
        if attr == "blind_ignore_mask":
            d = ignore_from_dead(d, int(getattr(args, "blind_dead_ignore_dilate", 2)))
        m = d[None].astype(np.float32)
    return resize_dead_mask(m, h, w, any_pixel=(attr == "blind_ignore_mask"))


def stitch_camera_mask(cam_front, cam_back, hw, args, attr="blind_dead_mask", device="cuda"):
    """HxW if ``cam_back is None``, else forward/back stitch Hx(2W). None if unavailable."""
    h, w = hw
    front = get_camera_mask(cam_front, h, w, args, attr)
    if cam_back is None:
        return None if front is None else front.to(device)
    back = get_camera_mask(cam_back, h, w, args, attr)
    if front is None and back is None:
        return None
    zeros = torch.zeros((1, h, w))
    front = zeros if front is None else front
    back = zeros if back is None else back
    breaks = (0, w // 2, 3 * w // 2, w * 2)
    out = torch.zeros((1, h, w * 2))
    out[:, :, breaks[1]:breaks[2]] = front
    out[:, :, breaks[2]:breaks[3]] = back[:, :, 0:(breaks[3] - breaks[2])]
    out[:, :, breaks[0]:breaks[1]] = back[:, :, (w - breaks[1] + breaks[0]):w]
    return out.to(device)


def force_dead_drop(depth, intensity, raydrop, dead):
    """Inference: dead pixels always drop (raydrop=1, depth/intensity=0)."""
    if dead is None or raydrop is None:
        return depth, intensity, raydrop
    d = dead.to(device=raydrop.device, dtype=raydrop.dtype)
    if d.shape[-2:] != raydrop.shape[-2:]:
        d = F.interpolate(d.unsqueeze(0), size=raydrop.shape[-2:], mode="nearest").squeeze(0)
    keep = 1.0 - d
    raydrop = torch.maximum(raydrop, d)
    if depth is not None:
        depth = depth * keep
    if intensity is not None:
        intensity = intensity * keep
    return depth, intensity, raydrop
