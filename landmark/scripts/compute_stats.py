#!/usr/bin/env python3
"""Compute the dataset statistics DiMP needs, from the official train split only.

  * vertex groups: recursive bisection of the mean (normalised) face shape;
  * point_std / center_std / motion_std: scales that bring points, tube centers
    and inter-frame displacements to ~unit variance, so the DDPM noise
    (unit variance) is meaningful for each of them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import blosc2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dimp.data import LandmarkDataset, prepare_mesh  # noqa: E402
import json  # noqa: E402

from dimp import regions  # noqa: E402
from dimp.tubes import TubeLayout, anatomical_groups, build_groups, motion_groups  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clips", type=int, default=2000)
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--tube-frames", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "stats/tube_stats.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grouping", choices=("bisect", "anatomical", "motion"), default="bisect",
        help="bisect: blind geometric split into --num-groups; anatomical: one group per"
        " annotated facial part (--num-groups is ignored); motion: recursive spectral split of"
        " the displacement-correlation graph into --num-groups",
    )
    args = parser.parse_args()

    data = ROOT / "data"
    dataset = LandmarkDataset(
        data / "landmark_2d_106/train",
        data / "meta/train-landmark_2d_106-meta.json",
        100,
        training=False,
        driver_split=data / "train_val_splits/driver_train_splits.json",
        source_split=data / "train_val_splits/source_train_splits.json",
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), min(args.num_clips, len(dataset)), replace=False)

    shapes, clips = [], []
    for index in indices:
        raw = torch.as_tensor(
            blosc2.load_tensor(str(dataset.root / dataset.meta[index]["relpath"])),
            dtype=torch.float32,
        )
        centered = raw - raw.mean(dim=1, keepdim=True)
        normalized = centered / centered.norm(dim=-1).mean().clamp_min(1e-6)
        shapes.append(normalized.mean(dim=0))
        clips.append(prepare_mesh(raw, 100, training=False)[..., :2])
    mean_shape = torch.stack(shapes).mean(dim=0)
    mesh = torch.stack(clips)  # (N, T, V, 2)

    if args.grouping == "anatomical":
        mapping = json.loads(regions.REGIONS_PATH.read_text())
        groups, valid = anatomical_groups(mean_shape, {int(k): v for k, v in mapping.items()})
        args.num_groups = groups.shape[0]
    elif args.grouping == "motion":
        groups, valid = motion_groups(mesh, args.num_groups)
    else:
        groups, valid = build_groups(mean_shape, args.num_groups)
    layout = TubeLayout(args.num_groups, groups.shape[1], args.tube_frames, 100)
    layout.set_groups(groups, valid)
    point_std = float(mesh.std())
    layout.scales[0] = point_std
    raw_tokens = layout.tokens(mesh)
    valid_points = layout.point_valid()
    center_std = float(raw_tokens["centers"].std())
    motion = raw_tokens["motion"]
    motion_valid = valid_points[None, :, :, None, None].expand_as(motion)
    motion_std = float(motion[motion_valid].std())
    layout.scales[1] = center_std
    layout.scales[2] = motion_std

    check = layout.tokens(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "groups": groups,
            "valid": valid,
            "scales": layout.scales.clone(),
            "num_groups": args.num_groups,
            "group_size": int(groups.shape[1]),
            "tube_frames": args.tube_frames,
            "num_clips": len(indices),
            "mean_shape": mean_shape,
            "grouping": args.grouping,
        },
        args.output,
    )
    print(f"groups: {args.num_groups} x n_max={groups.shape[1]}, sizes={valid.sum(1).tolist()}")
    print(f"point_std={point_std:.5f} center_std={center_std:.5f} motion_std={motion_std:.5f}")
    print(
        "normalised stds: points %.3f centers %.3f motion %.3f"
        % (
            float(check["points"][valid_points[None, :, :, None, None].expand_as(check["points"])].std()),
            float(check["centers"].std()),
            float(check["motion"][motion_valid].std()),
        )
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
