#!/usr/bin/env python3
"""Precompute one face extent per clip from the distributed 106-point landmarks.

For every mp4 (train, test-gen1, test-gen2) this records the min/max x/y of
all landmarks over all frames of the matching landmark file, in the video's
512x512 pixel coordinates. ``feat_diff.data`` turns that extent into a square
crop box (margin, and random jitter while training), so the video model sees
the face at about twice the resolution it had when the whole frame was
shrunk to 128x128. Only the distributed landmarks are used -- no external
face detector (competition rule).

Output: cache/face_boxes-<split>.npz with ``relpath`` (mp4 relpaths) and
``extent`` (N, 4) float32 = (x_min, y_min, x_max, y_max).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import blosc2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = {
    "train": ("meta/train-mp4-meta.json", "landmark_2d_106/train"),
    "test-gen1": ("meta/test_annon-mp4-gen1-meta.json", "landmark_2d_106/test-gen1"),
    "test-gen2": ("meta/test_annon-mp4-gen2-meta.json", "landmark_2d_106/test-gen2"),
}


def landmark_path(split: str, mp4_relpath: str) -> Path:
    stem = mp4_relpath.removesuffix(".mp4")
    if split == "train":  # d<driver>--t<source>/<clip>.mp4 -> .../<clip>/landmark_2d_106.bl2
        return DATA / SPLITS[split][1] / stem / "landmark_2d_106.bl2"
    return DATA / SPLITS[split][1] / f"{stem}.bl2"  # <uuid>.mp4 -> <uuid>.bl2


def extent(args: tuple[str, str]) -> list[float]:
    split, relpath = args
    points = np.asarray(blosc2.load_tensor(str(landmark_path(split, relpath))), dtype=np.float32)
    points = points.reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return [np.nan] * 4
    return [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    out_dir = ROOT / "cache"
    out_dir.mkdir(exist_ok=True)
    for split in args.splits:
        meta = json.loads((DATA / SPLITS[split][0]).read_text())
        relpaths = [item["relpath"] for item in meta]
        with Pool(args.workers) as pool:
            extents = pool.map(extent, [(split, r) for r in relpaths], chunksize=256)
        extents = np.asarray(extents, dtype=np.float32)
        missing = int(np.isnan(extents).any(axis=1).sum())
        size = np.median(extents[:, 2:] - extents[:, :2], axis=0)
        np.savez(out_dir / f"face_boxes-{split}.npz", relpath=np.asarray(relpaths), extent=extents)
        print(f"{split}: {len(relpaths)} clips, {missing} without landmarks, median face w x h = {size}", flush=True)


if __name__ == "__main__":
    main()
