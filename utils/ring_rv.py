#
# Ring-indexed range view (ring RV) for multi-beam blind lidars (QT128).
#
# Geometry (depth / intensity / chamfer) stays supervised on the uniform
# angular grid with the motion-compensated cloud. Raydrop, the sampling
# pattern and the output point cloud live on the ring RV: one row per laser
# ring (sorted by elevation, top row = highest), columns at that ring's native
# azimuth step. QT128 fires its lower 64 rings every 0.8° and the upper 64
# every 0.4°; on a 0.4° uniform grid half of the lower cells are never fired
# into, and deskew smears near rings over several uniform rows. On the ring RV
# an empty slot is a real dropout.
#
# A slot of a k-step ring spans k columns; it is stored at its first column
# (``slot_valid``) and sampled at the slot centre. The other k-1 columns are
# placeholders: never supervised, never emitted.
#
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LUT_NAME = "ring_lut.npz"
DEAD_NAME = "ring_dead_mask.npz"


def ring_rv_enabled(args):
    return bool(getattr(args, "ring_rv", False))


def _angles_deg(points_cam):
    """Elevation / azimuth (deg) and range in the RV camera frame (-y up, z forward)."""
    p = np.asarray(points_cam, dtype=np.float64)
    r = np.linalg.norm(p, axis=1)
    elev = 90.0 - np.degrees(np.arctan2(np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2), -p[:, 1]))
    az = np.degrees(np.arctan2(p[:, 0], p[:, 2]))
    return elev, az, r


# ---------------------------------------------------------------- LUT

def build_ring_lut(cam_infos, n_rings=128, min_range=4.0):
    """Per-sensor ring elevation / azimuth step from the (metric) training clouds.

    Far points (> ``min_range`` m) set the elevation: deskew moves near points
    by several degrees but barely moves far ones.
    """
    elev_far, elev_any, steps = defaultdict(list), defaultdict(list), defaultdict(list)
    sensors = set()
    for cam in cam_infos:
        rings = getattr(cam, "ring_ids", None)
        pts = cam.pointcloud_camera
        if rings is None or pts is None or len(pts) == 0:
            continue
        s = int(cam.sensor_id)
        sensors.add(s)
        el, az, r = _angles_deg(pts)
        order = np.argsort(rings, kind="stable")
        bounds = np.flatnonzero(np.diff(rings[order])) + 1
        for idx in np.split(order, bounds):
            ring = int(rings[idx[0]])
            if ring < 0 or ring >= n_rings:
                continue
            far = idx[r[idx] > min_range]
            if len(far) >= 5:
                elev_far[(s, ring)].append(float(np.median(el[far])))
            elif len(idx) >= 3:
                elev_any[(s, ring)].append(float(np.median(el[idx])))
            a = np.sort(az[idx])
            d = np.diff(a)
            d = d[(d > 0.05) & (d < 3.0)]
            if len(d) >= 10:
                steps[(s, ring)].append(float(np.median(d)))
    if not sensors:
        raise RuntimeError("ring_rv: no ring ids on any camera (PCD needs a `ring` field)")

    n_sensors = max(sensors) + 1
    elev = np.full((n_sensors, n_rings), np.nan)
    step = np.full((n_sensors, n_rings), np.nan)
    for s in sensors:
        for ring in range(n_rings):
            src = elev_far.get((s, ring)) or elev_any.get((s, ring))
            if src:
                elev[s, ring] = np.median(src)
            if steps.get((s, ring)):
                step[s, ring] = np.median(steps[(s, ring)])
        known = np.flatnonzero(np.isfinite(elev[s]))
        if len(known) == 0:
            continue
        missing = np.flatnonzero(~np.isfinite(elev[s]))
        if len(missing):
            elev[s, missing] = np.interp(missing, known, elev[s, known])
            # Rings never seen past either end (usually body-blocked bottom rings):
            # extrapolate from the nearest known rings instead of stacking on the edge.
            for side, ref in (("low", known[:5]), ("high", known[-5:])):
                out = missing[missing < known[0]] if side == "low" else missing[missing > known[-1]]
                if len(out) and len(ref) >= 2:
                    a, b = np.polyfit(ref, elev[s, ref], 1)
                    elev[s, out] = a * out + b
        known_step = np.flatnonzero(np.isfinite(step[s]))
        for ring in np.flatnonzero(~np.isfinite(step[s])):
            # Same step as the ring closest in elevation.
            if len(known_step):
                j = known_step[np.argmin(np.abs(elev[s, known_step] - elev[s, ring]))]
                step[s, ring] = step[s, j]
            else:
                step[s, ring] = 0.4
    elev = np.nan_to_num(elev, nan=0.0)
    step = np.nan_to_num(step, nan=0.4)
    row_ring = np.argsort(-elev, axis=1, kind="stable")
    ring_to_row = np.argsort(row_ring, axis=1)
    return {
        "elev_deg": elev.astype(np.float32),
        "step_deg": step.astype(np.float32),
        "row_ring": row_ring.astype(np.int64),
        "ring_to_row": ring_to_row.astype(np.int64),
        "n_rings": np.int64(n_rings),
    }


def save_ring_lut(path, lut):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **lut)


def load_ring_lut(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


_LUT = {}
_GEOM = {}


def _lut_path(args):
    return os.path.join(args.model_path, LUT_NAME)


def set_ring_lut(args, lut):
    _LUT[_lut_path(args)] = lut
    _GEOM.clear()


def get_ring_lut(args):
    path = _lut_path(args)
    if path not in _LUT:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ring_rv: {path} missing (built by the loader)")
        _LUT[path] = load_ring_lut(path)
    return _LUT[path]


def n_ring_rows(args):
    return int(get_ring_lut(args)["n_rings"])


def describe_lut(lut):
    lines = []
    for s in range(lut["elev_deg"].shape[0]):
        el, st = lut["elev_deg"][s], lut["step_deg"][s]
        n08 = int((st > 0.6).sum())
        lines.append(f"[Ring RV] sensor {s}: elev {el.min():.1f}..{el.max():.1f} deg, "
                     f"{n08} rings @0.8deg / {len(st) - n08} @0.4deg, "
                     f"min gap {np.diff(np.sort(el)).min():.2f} deg")
    return "\n".join(lines)


# ---------------------------------------------------------------- slot geometry

class RingGeometry:
    """Slot layout of one sensor on an (h, w) render grid. All tensors on CPU."""

    def __init__(self, lut, sensor, h, w, vfov, hfov):
        R = int(lut["n_rings"])
        rings = lut["row_ring"][sensor]
        elev = lut["elev_deg"][sensor][rings].astype(np.float64)
        step = lut["step_deg"][sensor][rings].astype(np.float64)
        colw = (hfov[1] - hfov[0]) / w
        k = np.maximum(1, np.round(step / colw)).astype(np.int64)
        cols = np.arange(w)[None]
        base = (cols // k[:, None]) * k[:, None]
        colf = np.minimum(base + (k[:, None] - 1) / 2.0, w - 1).astype(np.float64)
        rowf = (vfov[1] - elev) / (vfov[1] - vfov[0]) * h
        in_fov = (rowf >= -0.5) & (rowf < h - 0.5)
        slot_valid = (cols % k[:, None] == 0) & in_fov[:, None]

        self.R, self.h, self.w, self.hfov = R, h, w, hfov
        self.k = k
        self.ring_to_row = lut["ring_to_row"][sensor]
        self.elev = elev
        self.slot_valid = torch.from_numpy(slot_valid[None].astype(np.float32))

        # Bilinear taps on the render grid.
        rowc = np.clip(rowf, 0, h - 1)
        r0 = np.floor(rowc).astype(np.int64)
        r1 = np.minimum(r0 + 1, h - 1)
        wr = (rowc - r0)[:, None].repeat(w, 1)
        c0 = np.floor(colf).astype(np.int64)
        c1 = np.minimum(c0 + 1, w - 1)
        wc = colf - c0
        r0b, r1b = r0[:, None].repeat(w, 1), r1[:, None].repeat(w, 1)
        self.taps = torch.from_numpy(np.stack([r0b * w + c0, r0b * w + c1, r1b * w + c0, r1b * w + c1]))
        self.weights = torch.from_numpy(np.stack(
            [(1 - wr) * (1 - wc), (1 - wr) * wc, wr * (1 - wc), wr * wc]).astype(np.float32))

        # Slot ray directions (camera frame, same convention as pano_to_lidar).
        theta = np.radians(90.0 - elev)[:, None].repeat(w, 1)
        phi = np.radians(hfov[0] + colf * colw)
        dirs = np.stack([np.sin(theta) * np.sin(phi), -np.cos(theta), np.sin(theta) * np.cos(phi)])
        self.dirs = torch.from_numpy(dirs.astype(np.float32))

    def project(self, points, rings, intensity=None):
        """Bin camera-frame points into ring slots (nearest return wins). (1,R,w) depth, intensity."""
        R, w = self.R, self.w
        depth = np.zeros((1, R, w), dtype=np.float32)
        inten = np.zeros((1, R, w), dtype=np.float32)
        if points is None or len(points) == 0 or rings is None:
            return depth, inten
        if intensity is None:
            intensity = np.ones(len(points), dtype=np.float32)
        _, az, r = _angles_deg(points)
        u = (az - self.hfov[0]) / ((self.hfov[1] - self.hfov[0]) / w)
        rings = np.asarray(rings).astype(np.int64)
        keep = (u >= -0.5) & (u < w - 0.5) & (rings >= 0) & (rings < R)
        if not keep.any():
            return depth, inten
        row = self.ring_to_row[rings[keep]]
        c = np.around(u[keep]).astype(np.int64)
        k = self.k[row]
        col = (c // k) * k
        flat = row * w + col
        rr = r[keep]
        order = np.lexsort((rr, flat))
        flat_s = flat[order]
        first = np.ones(len(flat_s), dtype=bool)
        first[1:] = flat_s[1:] != flat_s[:-1]
        sel = order[first]
        depth.reshape(-1)[flat[sel]] = rr[sel]
        inten.reshape(-1)[flat[sel]] = np.asarray(intensity)[keep][sel]
        depth *= self.slot_valid[0].numpy()
        inten *= self.slot_valid[0].numpy()
        return depth, inten


def get_geometry(args, sensor, h, w):
    key = (_lut_path(args), int(sensor), int(h), int(w))
    if key not in _GEOM:
        _GEOM[key] = RingGeometry(get_ring_lut(args), int(sensor), int(h), int(w),
                                  tuple(args.vfov), tuple(args.hfov))
    return _GEOM[key]


def sample_to_ring(maps, ref_depth, geom, rel_thresh=0.1):
    """Sample render-grid maps (C,H,W) at slot rays → (C,R,W).

    Bilinear where the four taps lie on one surface; nearest tap across depth
    edges or holes, so no flying points between foreground and background.
    """
    C = maps.shape[0]
    device = maps.device
    taps = geom.taps.to(device)
    wts = geom.weights.to(device)
    flat = maps.reshape(C, -1)
    with torch.no_grad():
        d = ref_depth.reshape(-1)[taps]  # (4,R,W)
        used = wts > 1e-6
        big = torch.where(used, d, torch.full_like(d, -1.0)).amax(0)
        small = torch.where(used, d, torch.full_like(d, float("inf"))).amin(0)
        edge = (big - small > rel_thresh * small.clamp_min(1e-8)) | (small <= 0)
        nearest = F.one_hot(wts.argmax(0), 4).permute(2, 0, 1).to(wts.dtype)
        wts = torch.where(edge[None], nearest, wts)
    out = (flat[:, taps] * wts[None]).sum(1)
    return out


def ring_to_points(depth, geom):
    """(1,R,W) slot depth → (N,3) camera-frame points along slot rays."""
    dirs = geom.dirs.to(depth.device)
    mask = depth[0] > 0
    return (dirs * depth)[:, mask].permute(1, 0)


def uniform_to_points(depth, vfov, hfov):
    from utils.graphics_utils import pano_to_lidar
    return pano_to_lidar(depth, vfov, hfov)


# ---------------------------------------------------------------- dead zone

def _key(sensor_id, towards):
    return f"{int(sensor_id or 0)}_{towards or 'forward'}"


def _keep_bottom_connected(dead):
    n, labels = cv2.connectedComponents(dead.astype(np.uint8), connectivity=8)
    if n <= 1:
        return dead
    bottom = np.unique(labels[-1][labels[-1] > 0])
    return np.isin(labels, bottom)


def compute_ring_dead_masks(cam_infos, args, thresh=0.02, bottom_connected=True):
    """{key: (R,W) bool dead}: slots no training frame ever hits (body / mount occlusion)."""
    w = int(args.hw[1])
    hits, counts = {}, {}
    for cam in cam_infos:
        geom = get_geometry(args, cam.sensor_id, int(args.hw[0]), w)
        k = _key(cam.sensor_id, cam.towards)
        d, _ = geom.project(cam.pointcloud_camera, getattr(cam, "ring_ids", None))
        occ = d[0] > 0
        body = getattr(cam, "ego_body_points_cam", None)
        if body is not None and len(body):
            db, _ = geom.project(body, getattr(cam, "ego_body_ring", None))
            occ |= db[0] > 0
        if k not in hits:
            hits[k] = np.zeros(occ.shape, dtype=np.float64)
            counts[k] = 0
        hits[k] += occ
        counts[k] += 1
    dead, freq = {}, {}
    for key, h in hits.items():
        geom = get_geometry(args, int(key.split("_")[0]), int(args.hw[0]), w)
        f = h / max(counts[key], 1)
        # Spread each slot's frequency over its placeholder columns.
        cols = np.arange(w)[None]
        base = (cols // geom.k[:, None]) * geom.k[:, None]
        f = np.take_along_axis(f, base, axis=1)
        dd = f < thresh
        if bottom_connected:
            dd = _keep_bottom_connected(dd)
        dead[key] = dd
        freq[key] = f.astype(np.float32)
    return dead, freq


def build_ring_dead_masks(train_cam_infos, args):
    cache = os.path.join(args.model_path, DEAD_NAME)
    R, w = n_ring_rows(args), int(args.hw[1])
    reuse = (args.test_only or getattr(args, "start_checkpoint", None)) and os.path.isfile(cache)
    dead = None
    if reuse:
        data = np.load(cache)
        if tuple(data["hw"]) == (R, w):
            dead = {n[5:]: data[n].astype(bool) for n in data.files if n.startswith("dead_")}
            print(f"[Ring dead] reused {cache}")
    if dead is None:
        dead, freq = compute_ring_dead_masks(
            train_cam_infos, args,
            thresh=float(getattr(args, "blind_dead_thresh", 0.02)),
            bottom_connected=bool(getattr(args, "blind_dead_bottom_connected", True)))
        payload = {"hw": np.asarray((R, w), dtype=np.int64)}
        for k in dead:
            payload[f"dead_{k}"] = dead[k]
            payload[f"freq_{k}"] = freq[k]
        np.savez_compressed(cache, **payload)
        print(f"[Ring dead] saved {cache}")
    from utils.blind_dead_mask import ignore_from_dead
    dilate = int(getattr(args, "blind_dead_ignore_dilate", 2))
    ignore = {k: ignore_from_dead(d, dilate) for k, d in dead.items()}
    for k in sorted(dead):
        print(f"[Ring dead] {k}: dead={dead[k].mean():.3f} ignore={ignore[k].mean():.3f} of ring RV")
    return dead, ignore


def attach_ring_dead_masks(cam_infos, dead, ignore):
    out = []
    for cam in cam_infos:
        k = _key(cam.sensor_id, cam.towards)
        if k not in dead:
            out.append(cam)
            continue
        out.append(cam._replace(
            ring_dead_mask=dead[k][None].astype(np.float32),
            ring_ignore_mask=ignore[k][None].astype(np.float32)))
    return out


def resize_ring_mask(mask, R, w, any_pixel):
    """Resize (1,R,W0) → (1,R,w) along columns only."""
    if mask is None:
        return None
    m = torch.from_numpy(np.asarray(mask)).float() if not torch.is_tensor(mask) else mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[-2:] == (R, w):
        return m
    area = F.interpolate(m.unsqueeze(0), size=(R, w), mode="area").squeeze(0)
    return (area > 0).float() if any_pixel else (area >= 0.5).float()


# ---------------------------------------------------------------- per-camera GT

def build_camera_ring_gt(args, cam_info, h, w):
    """Ring-RV GT for one camera at render grid (h, w). Returns a dict of (1,R,w) tensors."""
    geom = get_geometry(args, cam_info.sensor_id, h, w)
    R = geom.R
    depth, inten = geom.project(cam_info.pointcloud_camera, getattr(cam_info, "ring_ids", None),
                                cam_info.intensity)
    out = {
        "ring_depth": torch.from_numpy(depth),
        "ring_intensity": torch.from_numpy(inten),
        "ring_body_depth": None,
        "ring_body_intensity": None,
        "ring_ego_mask": None,
    }
    body = getattr(cam_info, "ego_body_points_cam", None)
    if body is not None and len(body):
        bd, bi = geom.project(body, getattr(cam_info, "ego_body_ring", None),
                              getattr(cam_info, "ego_body_intensity", None))
        out["ring_body_depth"] = torch.from_numpy(bd)
        out["ring_body_intensity"] = torch.from_numpy(bi)
        out["ring_ego_mask"] = torch.from_numpy(((bd > 0) & (depth == 0)).astype(np.float32))
    out["ring_dead_mask"] = resize_ring_mask(getattr(cam_info, "ring_dead_mask", None), R, w, False)
    out["ring_ignore_mask"] = resize_ring_mask(getattr(cam_info, "ring_ignore_mask", None), R, w, True)
    return out


def camera_ring_valid(cam, geom, device):
    """Slots that take raydrop loss / metrics: fired, not body-only, not dead-zone band."""
    valid = geom.slot_valid.to(device) > 0.5
    ego = getattr(cam, "ring_ego_mask", None)
    if ego is not None:
        valid = valid & (ego.to(device) <= 0.5)
    ign = getattr(cam, "ring_ignore_mask", None)
    if ign is not None:
        valid = valid & (ign.to(device) <= 0.5)
    return valid


def coord_channels(geom, sensor, n_sensors, device):
    """UNet side inputs: slot elevation, azimuth, sensor one-hot. (2+n_sensors, R, W)."""
    R, w = geom.R, geom.w
    el = torch.from_numpy((geom.elev / 90.0).astype(np.float32))[:, None].expand(R, w)
    az = torch.linspace(-1, 1, w)[None].expand(R, w)
    onehot = torch.zeros(n_sensors, R, w)
    onehot[int(sensor)] = 1
    return torch.cat([el[None], az[None], onehot]).to(device)
