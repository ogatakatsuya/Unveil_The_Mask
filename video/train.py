#!/usr/bin/env python3
"""Video branch: inter-frame differencing model (Chapariniya et al., IEEE FG 2026) trained from scratch."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from feat_diff.data import CrossIdentityBatchSampler, VideoClipDataset, load_face_extents
from feat_diff.model import FeatDiffModel, ModelConfig
from feat_diff.objective import cross_source_alignment_loss, supervised_contrastive_loss
from feat_diff.runtime import cosine_schedule, distributed_context, evaluate, save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("video")


@dataclass
class TrainConfig:
    data_root: Path = ROOT / "data"
    output_dir: Path = ROOT / "output/video"
    iterations: int = 10_000
    grad_accum_steps: int = 8  # micro-batches per optimizer step
    validation_interval: int = 500
    early_stopping_patience: int = 5
    log_interval: int = 100
    drivers_per_batch: int = 8
    sources_per_driver: int = 4
    validation_batch_size: int = 8
    workers: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    temperature: float = 0.07
    cross_source_loss_weight: float = 0.1
    crop_jitter: float = 0.05  # face box jitter during training (fraction of the box side)
    seed: int = 0


def build_loaders(config: TrainConfig, model_config: ModelConfig, rank: int) -> tuple[DataLoader, DataLoader]:
    data = config.data_root
    common = {
        "root": data / "mp4/train",
        "meta_path": data / "meta/train-mp4-meta.json",
        "clip_length": model_config.clip_length,
        "frame_size": model_config.frame_size,
        "face_extents": load_face_extents("train") if model_config.face_crop else None,
        "crop_margin": model_config.crop_margin,
    }
    train_dataset = VideoClipDataset(
        **common, crop_jitter=config.crop_jitter, training=True,
        driver_split=data / "train_val_splits/driver_train_splits.json",
        source_split=data / "train_val_splits/source_train_splits.json",
    )
    val_dataset = VideoClipDataset(
        **common, training=False,
        driver_split=data / "train_val_splits/driver_val_splits.json",
        source_split=data / "train_val_splits/source_val_splits.json",
    )
    sampler = CrossIdentityBatchSampler(
        train_dataset, config.drivers_per_batch, config.sources_per_driver,
        config.iterations * config.grad_accum_steps, config.seed, rank,
    )
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
    defaults, default_model = TrainConfig(), ModelConfig()
    parser = argparse.ArgumentParser(description="Train the video branch from scratch")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--diff-mode", choices=("feat", "pixel"), default=default_model.diff_mode,
                        help="feat: F(I[t+1]) - F(I[t]); pixel: F(I[t+1] - I[t])")
    parser.add_argument("--face-crop", action="store_true", help="landmark-guided face crop")
    parser.add_argument("--frame-size", type=int, default=default_model.frame_size)
    parser.add_argument("--clip-length", type=int, default=default_model.clip_length)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    config = TrainConfig(
        output_dir=args.output_dir, iterations=args.iterations, grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate, workers=args.workers, seed=args.seed,
    )
    model_config = ModelConfig(
        clip_length=args.clip_length, frame_size=args.frame_size, face_crop=args.face_crop, diff_mode=args.diff_mode,
    )

    rank, world_size, local_rank = distributed_context()
    seed_everything(config.seed + rank)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True,
        handlers=[logging.FileHandler(config.output_dir / "train.log")] if rank == 0 else [logging.NullHandler()],
    )
    train_dict = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    if rank == 0:
        (config.output_dir / "config.json").write_text(
            json.dumps({"pretrained": False, "model": model_config.to_dict(), "training": train_dict}, indent=2) + "\n"
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config, model_config, rank)
    model = FeatDiffModel(model_config).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = cosine_schedule(optimizer, config.warmup_steps, config.iterations)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_auc, no_improve = -1.0, 0

    accum = config.grad_accum_steps
    progress = tqdm(train_loader, total=config.iterations * accum, desc="video", unit="micro-batch",
                    dynamic_ncols=True, miniters=config.log_interval, disable=rank != 0)
    iteration = 0
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "identity": 0.0, "cross_source": 0.0}
    for micro_step, batch in enumerate(progress, start=1):
        model.train()
        frames = batch["frames"].to(device, non_blocking=True)
        driver_ids = batch["driver_id"].to(device, non_blocking=True)
        source_ids = batch["source_id"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            embeddings = model(frames)
            identity_loss = supervised_contrastive_loss(embeddings, driver_ids, config.temperature)
            source_loss = cross_source_alignment_loss(embeddings, driver_ids, source_ids)
            loss = identity_loss + config.cross_source_loss_weight * source_loss
        scaler.scale(loss / accum).backward()
        sums["loss"] += loss.item() / accum
        sums["identity"] += identity_loss.item() / accum
        sums["cross_source"] += source_loss.item() / accum
        if micro_step % accum:
            continue
        iteration += 1
        losses, sums = sums, {"loss": 0.0, "identity": 0.0, "cross_source": 0.0}
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0 and (iteration == 1 or iteration % config.log_interval == 0):
            LOGGER.info("iteration %d/%d | %s | lr %.3g", iteration, config.iterations,
                        " | ".join(f"{k} {v:.5f}" for k, v in losses.items()), scheduler.get_last_lr()[0])

        stop_signal = torch.zeros(1, dtype=torch.uint8, device=device)
        if iteration % config.validation_interval == 0 or iteration == config.iterations:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                metrics = evaluate(model.module if hasattr(model, "module") else model, val_loader, device)
                text = " | ".join(f"{key}={value:.5f}" for key, value in metrics.items())
                LOGGER.info("validation | %s", text)
                tqdm.write(f"validation at {iteration} | {text}")
                if metrics["mean_auc"] > best_auc:
                    best_auc, no_improve = metrics["mean_auc"], 0
                    save_checkpoint(config.output_dir / "best_model.pt", model, optimizer, scheduler, iteration,
                                    model_config.to_dict(), train_dict, metrics)
                else:
                    no_improve += 1
                    if no_improve >= config.early_stopping_patience:
                        tqdm.write(f"early stopping at {iteration} | best mean_auc={best_auc:.5f}")
                        stop_signal[0] = 1
            if dist.is_initialized():
                dist.broadcast(stop_signal, src=0)
                dist.barrier()
        if bool(stop_signal.item()):
            break

    progress.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
