#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import blosc2
import torch


def validate(path: Path) -> torch.Tensor:
    tensor = torch.as_tensor(blosc2.load_tensor(str(path)))
    if tensor.ndim != 2 or tensor.shape[0] != 40_200 or tensor.shape[1] < 1:
        raise ValueError(f"{path}: expected (40200, D), got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WBA2026 submission ZIP")
    parser.add_argument("gen1", type=Path)
    parser.add_argument("gen2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = validate(args.gen1), validate(args.gen2)
    if first.shape[1] != second.shape[1]:
        raise ValueError("gen1 and gen2 embedding dimensions differ")
    manifest = {
        "matching": "cosine",
        "gen1": args.gen1.name,
        "gen2": args.gen2.name,
        "modality": "landmarks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.gen1, args.gen1.name)
        archive.write(args.gen2, args.gen2.name)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    print(f"created {args.output}")


if __name__ == "__main__":
    main()

