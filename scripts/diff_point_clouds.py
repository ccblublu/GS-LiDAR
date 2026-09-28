#!/usr/bin/env python3
"""Compare two PCDs; keep points that differ (no neighbor within --tol).

Outputs:
  *_only_a.pcd  — in A but not near B
  *_only_b.pcd  — in B but not near A
  *_diff.pcd    — union, with source=0 (A) / 1 (B)

Example:
  python scripts/diff_point_clouds.py \\
    --a .../frame000_imu_body_orig.pcd \\
    --b .../frame000_imu_body_template.pcd \\
    --tol 0.05 \\
    --out_prefix .../frame000_body_diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_pcd(path: Path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            text = line.decode("ascii", errors="ignore").strip()
            header.append(text)
            if text.startswith("DATA"):
                break
        meta = {}
        for ln in header:
            if ln and not ln.startswith("#") and " " in ln:
                k, v = ln.split(" ", 1)
                meta[k] = v
        n = int(meta["POINTS"])
        fields = meta["FIELDS"].split()
        sizes = list(map(int, meta["SIZE"].split()))
        types = meta["TYPE"].split()
        raw = f.read()
        mode = meta["DATA"].split()[0]
    if mode != "binary":
        raise ValueError(f"Only binary PCD: {path}")
    dtype = []
    for name, sz, tp in zip(fields, sizes, types):
        dtype.append((name, {"F4": "<f4", "I4": "<i4", "U1": "<u1"}[tp + str(sz)]))
    arr = np.frombuffer(raw, dtype=np.dtype(dtype), count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    inten = (arr["intensity"].astype(np.float32)
             if "intensity" in arr.dtype.names else np.ones(n, np.float32))
    return xyz, inten


def write_pcd(path: Path, xyz, intensity=None, source=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    if intensity is None:
        intensity = np.ones(n, np.float32)
    if source is None:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["intensity"] = intensity.astype(np.float32)
        fields, size, types, count = "x y z intensity", "4 4 4 4", "F F F F", "1 1 1 1"
    else:
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("intensity", "<f4"), ("source", "<i4"),
        ])
        arr = np.empty(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["intensity"] = intensity.astype(np.float32)
        arr["source"] = source.astype(np.int32)
        fields, size, types, count = (
            "x y z intensity source", "4 4 4 4 4", "F F F F I", "1 1 1 1 1")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {fields}\nSIZE {size}\nTYPE {types}\nCOUNT {count}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def unmatched_mask(query, ref, tol, chunk=20000):
    """True where query has no ref neighbor within tol (Euclidean)."""
    if len(query) == 0:
        return np.zeros(0, dtype=bool)
    if len(ref) == 0:
        return np.ones(len(query), dtype=bool)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref.astype(np.float64))
        dist, _ = tree.query(query.astype(np.float64), k=1, workers=-1)
        return dist > tol
    except Exception:
        # fallback: chunked brute force
        unmatched = np.ones(len(query), dtype=bool)
        tol2 = tol * tol
        for i0 in range(0, len(query), chunk):
            q = query[i0:i0 + chunk].astype(np.float64)
            best = np.full(len(q), np.inf)
            for j0 in range(0, len(ref), chunk):
                r = ref[j0:j0 + chunk].astype(np.float64)
                d2 = ((q[:, None, :] - r[None, :, :]) ** 2).sum(-1).min(axis=1)
                best = np.minimum(best, d2)
            unmatched[i0:i0 + chunk] = best > tol2
        return unmatched


def main():
    parser = argparse.ArgumentParser(description="Keep differing points between two PCDs")
    parser.add_argument("--a", type=str, required=True, help="Point cloud A")
    parser.add_argument("--b", type=str, required=True, help="Point cloud B")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Match radius (m); larger = fewer 'diff' points")
    parser.add_argument("--out_prefix", type=str, required=True)
    args = parser.parse_args()

    xa, ia = read_pcd(Path(args.a))
    xb, ib = read_pcd(Path(args.b))
    print(f"[diff] A n={len(xa)}  B n={len(xb)}  tol={args.tol}m")

    only_a = unmatched_mask(xa, xb, args.tol)
    only_b = unmatched_mask(xb, xa, args.tol)
    print(f"[diff] only_A={int(only_a.sum())}  only_B={int(only_b.sum())}  "
          f"matched_A={int((~only_a).sum())} matched_B={int((~only_b).sum())}")

    prefix = Path(args.out_prefix)
    write_pcd(Path(str(prefix) + "_only_a.pcd"), xa[only_a], ia[only_a])
    write_pcd(Path(str(prefix) + "_only_b.pcd"), xb[only_b], ib[only_b])

    diff_xyz = np.concatenate([xa[only_a], xb[only_b]], axis=0)
    diff_i = np.concatenate([ia[only_a], ib[only_b]], axis=0)
    diff_s = np.concatenate([
        np.zeros(int(only_a.sum()), np.int32),
        np.ones(int(only_b.sum()), np.int32),
    ])
    write_pcd(Path(str(prefix) + "_diff.pcd"), diff_xyz, diff_i, diff_s)
    print(f"[diff] wrote {prefix}_only_a.pcd / _only_b.pcd / _diff.pcd "
          f"(source 0=A, 1=B)")


if __name__ == "__main__":
    main()
