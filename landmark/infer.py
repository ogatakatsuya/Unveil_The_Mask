#!/usr/bin/env python3
"""Test-set embeddings of one landmark model (Stage 2 checkpoint)."""
from __future__ import annotations

import argparse
from pathlib import Path

import blosc2
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dimp.data import LandmarkDataset
from dimp.model import DiMPConfig, DiMPIdentity

ROOT = Path(__file__).resolve().parent


def build_dataset(generation: str, model_config: DiMPConfig) -> LandmarkDataset:
    return LandmarkDataset(
        root=ROOT / f"data/landmark_2d_106/test-{generation}",
        meta_path=ROOT
        / f"data/meta/test_annon-landmark_2d_106-{generation}-meta.json",
        frames=model_config.frames,
        training=False,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DiMPIdentity, DiMPConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("external_pretrained") is not False:
        raise RuntimeError("checkpoint does not declare external_pretrained=False")
    model_config = DiMPConfig.from_dict(checkpoint["model_config"])
    model = DiMPIdentity(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, model_config


@torch.no_grad()
def run_shard(
    model: DiMPIdentity,
    device: torch.device,
    dataset: LandmarkDataset,
    indices: range,
    batch_size: int,
    workers: int,
    desc: str,
) -> torch.Tensor:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    outputs = []
    for batch in tqdm(loader, desc=desc):
        mesh = batch["mesh"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs.append(model(mesh).cpu())
    return torch.cat(outputs).float().contiguous()


def shard_ranges(total: int, num_shards: int) -> list[range]:
    base, remainder = divmod(total, num_shards)
    ranges = []
    start = 0
    for shard in range(num_shards):
        size = base + (1 if shard < remainder else 0)
        ranges.append(range(start, start + size))
        start += size
    return ranges


def worker(
    rank: int,
    checkpoint_path: Path,
    generation: str,
    batch_size: int,
    workers: int,
    ranges: list[range],
    shared: torch.Tensor,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model, model_config = load_model(checkpoint_path, device)
    dataset = build_dataset(generation, model_config)
    indices = ranges[rank]
    if len(indices) == 0:
        return
    embeddings = run_shard(
        model, device, dataset, indices, batch_size, workers, f"{generation} gpu{rank}"
    )
    shared[indices.start : indices.stop] = embeddings


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WBA embeddings")
    parser.add_argument("--generation", choices=("gen1", "gen2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )
    args = parser.parse_args()

    output = args.output or ROOT / "output/submission" / f"embeddings-{args.generation}.bl2"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.num_gpus <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_config = load_model(args.checkpoint, device)
        dataset = build_dataset(args.generation, model_config)
        embeddings = run_shard(
            model,
            device,
            dataset,
            range(len(dataset)),
            args.batch_size,
            args.workers,
            args.generation,
        )
    else:
        probe_model, model_config = load_model(args.checkpoint, torch.device("cpu"))
        del probe_model
        dataset_len = len(build_dataset(args.generation, model_config))
        ranges = shard_ranges(dataset_len, args.num_gpus)
        shared = torch.empty((dataset_len, model_config.embedding_dim), dtype=torch.float32)
        shared.share_memory_()
        mp.spawn(
            worker,
            args=(
                args.checkpoint,
                args.generation,
                args.batch_size,
                args.workers,
                ranges,
                shared,
            ),
            nprocs=args.num_gpus,
            join=True,
        )
        embeddings = shared

    blosc2.save_tensor(embeddings, str(output), mode="w")
    print(f"saved {tuple(embeddings.shape)} to {output}")


if __name__ == "__main__":
    main()

