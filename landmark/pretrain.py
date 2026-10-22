#!/usr/bin/env python3
"""Stage 1: diffusion masked pretraining on landmark tokens (training split, no labels)."""
from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dimp.data import LandmarkDataset, RandomBatchSampler
from dimp.model import DiMPConfig, DiMPPretrainer
from dimp.runtime import cosine_schedule, distributed_context, evaluate, evaluate_pretext, save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("pretrain")


@dataclass
class PretrainConfig:
    data_root: Path = ROOT / "data"
    stats: Path = ROOT / "stats/tube_stats.pt"
    output_dir: Path = ROOT / "output/pretrain"
    iterations: int = 25_000
    batch_size: int = 64  # per GPU and micro-batch
    grad_accum_steps: int = 3  # micro-batches per optimizer step
    validation_interval: int = 1000
    log_interval: int = 100
    validation_batch_size: int = 32
    workers: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    seed: int = 0


def build_loaders(config: PretrainConfig, rank: int, micro_batches: int) -> tuple[DataLoader, DataLoader]:
    data = config.data_root
    common = {"root": data / "landmark_2d_106/train", "meta_path": data / "meta/train-landmark_2d_106-meta.json", "frames": 100}
    train_dataset = LandmarkDataset(
        **common, training=True,
        driver_split=data / "train_val_splits/driver_train_splits.json",
        source_split=data / "train_val_splits/source_train_splits.json",
    )
    val_dataset = LandmarkDataset(
        **common, training=False,
        driver_split=data / "train_val_splits/driver_val_splits.json",
        source_split=data / "train_val_splits/source_val_splits.json",
    )
    sampler = RandomBatchSampler(len(train_dataset), config.batch_size, micro_batches, config.seed, rank)
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=config.workers,
        pin_memory=True, persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.validation_batch_size, shuffle=False, num_workers=config.workers,
        pin_memory=True, persistent_workers=config.workers > 0,
    )
    return train_loader, val_loader


def main() -> None:
    defaults, model_defaults = PretrainConfig(), DiMPConfig()
    parser = argparse.ArgumentParser(description="Stage 1: diffusion masked pretraining")
    parser.add_argument("--stats", type=Path, default=defaults.stats,
                        help="token grouping and scales from scripts/compute_stats.py (Geo, Anat or Motion)")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--validation-interval", type=int, default=defaults.validation_interval)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--lambda-mot", type=float, default=model_defaults.lambda_mot)
    parser.add_argument("--recon-target", choices=("position", "velocity"), default=model_defaults.recon_target)
    parser.add_argument("--motion-cond-dropout", type=float, default=model_defaults.motion_cond_dropout)
    parser.add_argument("--pseudo-gen", action="store_true", help="pseudo-generator randomization of the input")
    parser.add_argument("--pseudo-gen-dro-eta", type=float, default=model_defaults.pseudo_gen_dro_eta,
                        help="GroupDRO step size over the pseudo-generator families (0 = uniform)")
    parser.add_argument("--aux-tasks", default="motion_energy,region_motion,eye_shape",
                        help="comma-separated auxiliary tasks ('' for none)")
    parser.add_argument("--aux-weight", type=float, default=0.01, help="weight w of every auxiliary task")
    args = parser.parse_args()

    config = PretrainConfig(
        stats=args.stats, output_dir=args.output_dir, iterations=args.iterations, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, validation_interval=args.validation_interval, workers=args.workers,
        seed=args.seed,
    )
    stats = torch.load(config.stats, map_location="cpu")
    model_config = DiMPConfig(
        tube_frames=stats["tube_frames"], num_groups=stats["num_groups"], group_size=stats["group_size"],
        lambda_mot=args.lambda_mot, recon_target=args.recon_target, motion_cond_dropout=args.motion_cond_dropout,
        pseudo_gen=args.pseudo_gen, pseudo_gen_dro_eta=args.pseudo_gen_dro_eta,
        aux_tasks=tuple(t for t in args.aux_tasks.split(",") if t),
        motion_energy_aux_weight=args.aux_weight, region_motion_aux_weight=args.aux_weight,
        eye_shape_aux_weight=args.aux_weight,
    )

    rank, world_size, local_rank = distributed_context()
    seed_everything(config.seed + rank)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True,
        handlers=[logging.FileHandler(config.output_dir / "train.log")] if rank == 0 else [logging.NullHandler()],
    )
    serializable = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    if rank == 0:
        (config.output_dir / "config.json").write_text(
            json.dumps({"external_pretrained": False, "model": model_config.to_dict(), "pretrain": serializable}, indent=2) + "\n"
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    accum_steps = max(1, config.grad_accum_steps)
    train_loader, val_loader = build_loaders(config, rank, config.iterations * accum_steps)
    pretrainer = DiMPPretrainer(model_config)
    pretrainer.core.layout.set_groups(stats["groups"], stats["valid"])
    pretrainer.core.layout.scales.copy_(stats["scales"])
    pretrainer.to(device)
    model = DDP(pretrainer, device_ids=[local_rank]) if world_size > 1 else pretrainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = cosine_schedule(optimizer, config.warmup_steps, config.iterations)
    use_amp = device.type == "cuda"

    progress = tqdm(train_loader, total=config.iterations * accum_steps, desc="stage 1", unit="micro-batch",
                    dynamic_ncols=True, miniters=config.log_interval, disable=rank != 0)
    no_sync = model.no_sync if hasattr(model, "no_sync") else nullcontext
    optimizer.zero_grad(set_to_none=True)
    accumulated: dict[str, float] = {}
    iteration = 0
    for micro_step, batch in enumerate(progress, start=1):
        model.train()
        mesh = batch["mesh"].to(device, non_blocking=True)
        is_last_micro_step = micro_step % accum_steps == 0
        with no_sync() if not is_last_micro_step else nullcontext():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                losses = model(mesh)
            (losses["loss"] / accum_steps).backward()
        for key, value in losses.items():
            accumulated[key] = accumulated.get(key, 0.0) + float(value) / accum_steps
        if not is_last_micro_step:
            continue
        iteration += 1
        losses, accumulated = accumulated, {}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if rank == 0 and (iteration == 1 or iteration % config.log_interval == 0 or iteration == config.iterations):
            LOGGER.info("iteration %d/%d | %s | lr %.3g", iteration, config.iterations,
                        " | ".join(f"{k} {v:.5f}" for k, v in losses.items()), scheduler.get_last_lr()[0])

        if iteration % config.validation_interval == 0 or iteration == config.iterations:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                pretrainer.eval()
                metrics = evaluate_pretext(pretrainer, val_loader, device)
                probe = evaluate(pretrainer.pooled_features, val_loader, device)
                metrics.update({f"probe_{key}": value for key, value in probe.items()})
                text = " | ".join(f"{key}={value:.5f}" for key, value in metrics.items())
                LOGGER.info("validation | %s", text)
                tqdm.write(f"validation at {iteration} | {text}")
                save_checkpoint(config.output_dir / "last_core.pt", {
                    "iteration": iteration, "core": pretrainer.core.state_dict(), "model_config": model_config.to_dict(),
                    "pretrain_config": serializable, "metrics": metrics, "external_pretrained": False,
                })
            if dist.is_initialized():
                dist.barrier()

    progress.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
