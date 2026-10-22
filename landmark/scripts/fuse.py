#!/usr/bin/env python3
"""Fuse member embeddings into a submission ZIP, with its own members for each test set.

Gen 1 (Same Gen) and Gen 2 (Diff Gen) are scored independently, so each file
may use a different ensemble. A file's embedding concatenates the L2-normalized
member embeddings, landmark members scaled by sqrt(alpha / k) and video members
by sqrt((1 - alpha) / m), so that its cosine similarity is the weighted mean of
the members' cosine similarities. Member embeddings are read from
output/submission-<tag>/embeddings-<gen>.bl2 (landmark) and
../video/output/submission-<tag>/embeddings-<gen>.bl2 (video), as written by infer.py.

    python scripts/fuse.py --gen1-landmark <tags> --gen1-video <tags> --gen1-alpha 0.5 \
        --gen2-landmark <tags> --gen2-alpha 1.0 --output submission.zip
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path

import blosc2
import torch

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT.parent / "video"


def load(tag: str, gen: str, video: bool) -> torch.Tensor:
    base = (VIDEO_ROOT / "output" if video else ROOT / "output") / f"submission-{tag}" / f"embeddings-{gen}.bl2"
    x = torch.as_tensor(blosc2.load_tensor(str(base))).float()
    return torch.nn.functional.normalize(x, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    for g in ("gen1", "gen2"):
        parser.add_argument(f"--{g}-landmark", nargs="*", default=[])
        parser.add_argument(f"--{g}-video", nargs="*", default=[])
        parser.add_argument(f"--{g}-alpha", type=float, required=True, help="landmark weight a for this file")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    counts = [len(getattr(args, f"{g}_landmark")) + len(getattr(args, f"{g}_video")) for g in ("gen1", "gen2")]
    width = load((args.gen1_landmark or args.gen2_landmark)[0], "gen1", video=False).shape[1]
    dim = max(counts) * width
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for gen in ("gen1", "gen2"):
            lm, vid, a = getattr(args, f"{gen}_landmark"), getattr(args, f"{gen}_video"), getattr(args, f"{gen}_alpha")
            if not vid:
                a = 1.0
            if not lm:
                a = 0.0
            # each file packs only its own members (the files are scored independently,
            # so block positions need not match); zero-padded below to a common dimension
            blocks = [load(tag, gen, video=False) * math.sqrt(a / len(lm)) for tag in lm]
            blocks += [load(tag, gen, video=True) * math.sqrt((1 - a) / len(vid)) for tag in vid]
            z = torch.cat(blocks, dim=1)
            if z.shape[1] < dim:
                z = torch.cat([z, torch.zeros(z.shape[0], dim - z.shape[1])], dim=1)
            norms = z.norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), (gen, norms.min(), norms.max())
            path = Path(tmp) / f"embeddings-{gen}.bl2"
            blosc2.save_tensor(z.to(getattr(torch, args.dtype)).contiguous(), str(path), mode="w")  # a torch tensor: the scorer calls .float() on what load_tensor returns
            files[gen] = path
            print(f"{gen}: {tuple(z.shape)} landmark={lm} video={vid} a={a}")
        manifest = {"matching": "cosine", "gen1": "embeddings-gen1.bl2", "gen2": "embeddings-gen2.bl2", "modality": "landmarks"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for gen, path in files.items():
                zf.write(path, f"embeddings-{gen}.bl2")
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    record = {g: {"landmark": getattr(args, f"{g}_landmark"), "video": getattr(args, f"{g}_video"),
                  "alpha": getattr(args, f"{g}_alpha")} for g in ("gen1", "gen2")}
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(f"created {args.output} ({args.output.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
