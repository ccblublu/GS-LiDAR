#!/usr/bin/env python3
"""Denoise an ego-body IMU cloud via multi-frame voxel voting, then densify/mesh.

Stable vehicle returns occupy the same IMU voxels across frames; transient
scene noise only lights a voxel in a few frames. After voting we optionally
build a Poisson mesh (Open3D) and re-sample a dense surface cloud.

Example:
  python scripts/refine_ego_body_cloud.py \\
    --aabb -1.4 1.4 -2 2 -1.8 0 \\
    --vote_ratio 0.15 \\
    --out_dir eval_output/gacrt_reconstruction/PCI_20260825_qt128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gacrt_loader import load_lidar_xyz_intensity, points_in_aabb

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "build_ego_body_cloud", ROOT / "scripts" / "build_ego_body_cloud.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)
DEFAULT_AABB = _build.DEFAULT_AABB
parse_aabb = _build.parse_aabb
resolve_lidar_path = _build.resolve_lidar_path
write_pcd_binary = _build.write_pcd_binary


def write_ply_mesh(path: Path, vertices, triangles):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(v)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(t)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for p in v:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for face in t:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def accumulate_votes(frames, data_root, aabb, source_ids, voxel):
    """Return (mins, dims, votes, inten_sum, count, n_frames_used)."""
    mins = aabb[:, 0].astype(np.float64)
    maxs = aabb[:, 1].astype(np.float64)
    dims = np.floor((maxs - mins) / voxel).astype(np.int64) + 1
    votes = np.zeros(dims, dtype=np.uint16)
    inten_sum = np.zeros(dims, dtype=np.float64)
    hit_count = np.zeros(dims, dtype=np.uint32)
    n_used = 0

    for fr in frames:
        path = resolve_lidar_path(fr["lidar_path"], data_root)
        n_feats = int(fr.get("num_pts_feats", 6))
        xyz, inten, source = load_lidar_xyz_intensity(
            str(path), n_feats=n_feats, return_source=True)
        finite = np.isfinite(xyz).all(axis=1)
        xyz, inten = xyz[finite], inten[finite]
        if source is not None:
            source = source[finite]
        if source_ids is not None:
            if source is None:
                raise RuntimeError(f"No source channel in {path}")
            sel = np.isin(source, np.asarray(source_ids, dtype=np.int32))
            xyz, inten = xyz[sel], inten[sel]
        inside = points_in_aabb(xyz, aabb)
        xyz, inten = xyz[inside], inten[inside]
        if len(xyz) == 0:
            continue

        ix = np.floor((xyz.astype(np.float64) - mins) / voxel).astype(np.int64)
        ix = np.clip(ix, 0, dims - 1)
        flat = np.ravel_multi_index(ix.T, dims)
        # One vote per voxel per frame (presence), intensity averaged over hits.
        uniq, inv = np.unique(flat, return_inverse=True)
        votes.ravel()[uniq] += 1
        # mean intensity per unique voxel this frame, then accumulate
        inten_f = inten.astype(np.float64)
        sum_i = np.bincount(inv, weights=inten_f, minlength=len(uniq))
        cnt_i = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        mean_i = sum_i / np.maximum(cnt_i, 1.0)
        inten_sum.ravel()[uniq] += mean_i
        hit_count.ravel()[uniq] += 1
        n_used += 1

    return mins, dims, votes, inten_sum, hit_count, n_used


def votes_to_cloud(mins, dims, votes, inten_sum, hit_count, voxel, min_votes):
    mask = votes >= min_votes
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32))
    xyz = coords.astype(np.float64) * voxel + mins + 0.5 * voxel
    flat = np.ravel_multi_index(coords.T, dims)
    votes_f = votes.ravel()[flat].astype(np.int32)
    inten = inten_sum.ravel()[flat] / np.maximum(hit_count.ravel()[flat], 1)
    return xyz.astype(np.float32), inten.astype(np.float32), votes_f


def densify_grid(xyz, inten, voxel, subdiv):
    """Subdivide each occupied voxel only — volume stays inside voted support."""
    subdiv = max(int(subdiv), 1)
    if subdiv == 1:
        return xyz.astype(np.float32), inten.astype(np.float32)

    origin = xyz.min(0).astype(np.float64) - 0.5 * voxel
    ijk = np.round((xyz.astype(np.float64) - (origin + 0.5 * voxel)) / voxel).astype(np.int64)
    keys = np.ascontiguousarray(ijk).view(np.dtype((np.void, ijk.dtype.itemsize * 3)))
    _, uniq_idx = np.unique(keys, return_index=True)
    ijk = ijk[uniq_idx]
    inten_u = inten[uniq_idx]

    offs = (np.arange(subdiv) + 0.5) / subdiv
    gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
    local = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    parts_xyz, parts_i = [], []
    for cell, iv in zip(ijk, inten_u):
        base = origin + cell.astype(np.float64) * voxel
        parts_xyz.append(base + local * voxel)
        parts_i.append(np.full(len(local), float(iv), dtype=np.float64))
    return (np.concatenate(parts_xyz, axis=0).astype(np.float32),
            np.concatenate(parts_i, axis=0).astype(np.float32))


def densify_ball_pivot(xyz, voxel, out_prefix: Path):
    """Tight Ball-Pivoting mesh — preserves detail, does not inflate like Poisson."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(voxel * 3.0, 0.06), max_nn=30))
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.5]))

    radii = o3d.utility.DoubleVector(
        [voxel * 0.9, voxel * 1.4, voxel * 2.0, voxel * 2.8])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh_path = Path(str(out_prefix) + "_mesh_bp.ply")
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"[refine] ball-pivot mesh V={len(mesh.vertices)} "
          f"F={len(mesh.triangles)} -> {mesh_path}")
    if len(mesh.triangles) == 0:
        return mesh_path, None

    n_sample = min(max(len(xyz) * 20, 50_000), 500_000)
    dense = mesh.sample_points_uniformly(number_of_points=int(n_sample))
    dense_xyz = np.asarray(dense.points, dtype=np.float32)
    dense_path = Path(str(out_prefix) + "_dense_bp.pcd")
    write_pcd_binary(dense_path, dense_xyz, np.ones(len(dense_xyz), np.float32), None)
    print(f"[refine] ball-pivot dense n={len(dense_xyz)} -> {dense_path}")
    return mesh_path, dense_path


def main():
    parser = argparse.ArgumentParser(description="Vote-denoise + densify ego body cloud")
    parser.add_argument(
        "--transforms", type=str,
        default="data/PCI_20260825/transforms_PCI_20260825_all.json")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--n_frames", type=int, default=0,
                        help="0 = all frames in transforms")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--aabb", type=float, nargs=6, default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--sources", type=str, default="1,2,3,4")
    parser.add_argument("--voxel", type=float, default=0.03)
    parser.add_argument("--vote_ratio", type=float, default=0.15)
    parser.add_argument("--min_votes", type=int, default=0)
    parser.add_argument(
        "--from_voted", type=str, default=None,
        help="Skip re-voting; densify from an existing *_voted.npz")
    parser.add_argument(
        "--densify", type=str, default="grid",
        choices=["none", "grid", "ball_pivot", "grid+ball_pivot"],
        help="Densify mode. Poisson removed (inflates + oversmooths).")
    parser.add_argument("--subdiv", type=int, default=3,
                        help="Grid densify subdivision per axis (3 -> ~27x points)")
    parser.add_argument(
        "--out_dir", type=str,
        default="eval_output/gacrt_reconstruction/PCI_20260825_qt128")
    parser.add_argument("--tag", type=str, default="ego_body_imu_clean")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / args.tag

    if args.from_voted:
        data = np.load(args.from_voted)
        xyz = data["xyz"].astype(np.float32)
        inten = data["intensity"].astype(np.float32)
        voxel = float(data["voxel"]) if "voxel" in data.files else args.voxel
        print(f"[refine] loaded voted n={len(xyz)} voxel={voxel} from {args.from_voted}")
    else:
        meta = json.loads(Path(args.transforms).read_text())
        frames = [f for f in meta["frames"] if int(f.get("frame_idx", 0)) >= args.start_frame]
        if args.n_frames and args.n_frames > 0:
            frames = frames[: args.n_frames]
        aabb = parse_aabb(args.aabb)
        if args.aabb is None:
            aabb = np.asarray(DEFAULT_AABB, dtype=np.float64)
        if args.sources.strip().lower() == "all":
            source_ids = None
        else:
            source_ids = [int(x) for x in args.sources.split(",") if x.strip()]

        print(f"[refine] frames={len(frames)} voxel={args.voxel} aabb=\n{aabb}")
        mins, dims, votes, inten_sum, hit_count, n_used = accumulate_votes(
            frames, Path(args.data_root), aabb, source_ids, args.voxel)
        print(f"[refine] frames_with_hits={n_used} grid={dims} "
              f"occupied={(votes > 0).sum()}")

        min_votes = args.min_votes
        if min_votes <= 0:
            min_votes = max(1, int(np.ceil(args.vote_ratio * n_used)))
        print(f"[refine] min_votes={min_votes} (vote_ratio={args.vote_ratio})")
        for thr in sorted({1, 5, min_votes,
                           max(min_votes, int(0.3 * n_used)),
                           max(min_votes, int(0.5 * n_used))}):
            print(f"  votes>={thr}: {(votes >= thr).sum()} voxels")

        xyz, inten, vote_vals = votes_to_cloud(
            mins, dims, votes, inten_sum, hit_count, args.voxel, min_votes)
        print(f"[refine] voted cloud n={len(xyz)}")
        if len(xyz) == 0:
            raise SystemExit("Empty after voting — lower --vote_ratio")

        clean_pcd = Path(str(prefix) + "_voted.pcd")
        write_pcd_binary(clean_pcd, xyz, inten, None)
        np.savez_compressed(
            Path(str(prefix) + "_voted.npz"),
            xyz=xyz, intensity=inten, votes=vote_vals,
            voxel=np.float32(args.voxel), min_votes=np.int32(min_votes),
            n_frames_used=np.int32(n_used), aabb=aabb.astype(np.float32))
        print(f"[refine] wrote {clean_pcd} bbox={xyz.min(0)} .. {xyz.max(0)}")
        voxel = args.voxel

    mode = args.densify
    if mode in ("grid", "grid+ball_pivot"):
        dense_xyz, dense_i = densify_grid(xyz, inten, voxel, args.subdiv)
        dense_path = Path(str(prefix) + "_dense.pcd")
        write_pcd_binary(dense_path, dense_xyz, dense_i, None)
        np.savez_compressed(
            Path(str(prefix) + "_dense.npz"),
            xyz=dense_xyz, intensity=dense_i,
            voxel=np.float32(voxel / max(args.subdiv, 1)))
        print(f"[refine] grid densify x{args.subdiv}: "
              f"{len(xyz)} -> {len(dense_xyz)} bbox={dense_xyz.min(0)}..{dense_xyz.max(0)}")
        print(f"[refine] wrote {dense_path}")
    if mode in ("ball_pivot", "grid+ball_pivot"):
        densify_ball_pivot(xyz, voxel, prefix)

    print("[refine] done")


if __name__ == "__main__":
    main()
