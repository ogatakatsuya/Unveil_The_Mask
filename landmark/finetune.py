#!/usr/bin/env python3
"""Stage 2: driver-aware contrastive probing (frozen Stage 1 encoder, mean pooling, one linear layer)."""
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

from dimp.data import CrossIdentityBatchSampler, LandmarkDataset
from dimp.model import DiMPConfig, DiMPIdentity
from dimp.objective import supervised_contrastive_loss
from dimp.runtime import cosine_schedule, distributed_context, evaluate, save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("finetune")


@dataclass
class ProbeConfig:
    data_root: Path = ROOT / "data"
    output_dir: Path = ROOT / "output/probe"
    init_from: str = ""
    iterations: int = 5_000
    grad_accum_steps: int = 3
    validation_interval: int = 250
    early_stopping_patience: int = 8
    log_interval: int = 100
    drivers_per_batch: int = 8
    sources_per_driver: int = 4
    validation_batch_size: int = 32
    workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    temperature: float = 0.07
    seed: int = 0
    # apply the pseudo-generator transforms (dimp/pseudogen.py) to the training input
    pseudo_gen_aug: bool = False


def build_loaders(config: ProbeConfig, rank: int, micro_batches: int) -> tuple[DataLoader, DataLoader]:
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
    sampler = CrossIdentityBatchSampler(
        train_dataset, config.drivers_per_batch, config.sources_per_driver, micro_batches, config.seed, rank
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
    defaults = ProbeConfig()
    parser = argparse.ArgumentParser(description="Stage 2: contrastive linear probe on the frozen encoder")
    parser.add_argument("--init-from", type=Path, required=True, help="Stage 1 checkpoint (last_core.pt)")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--pseudo-gen-aug", action="store_true",
                        help="apply the pseudo-generator transforms to the Stage 2 input")
    args = parser.parse_args()

    config = ProbeConfig(
        output_dir=args.output_dir, init_from=str(args.init_from), iterations=args.iterations,
        grad_accum_steps=args.grad_accum_steps, learning_rate=args.learning_rate, workers=args.workers,
        seed=args.seed, pseudo_gen_aug=args.pseudo_gen_aug,
    )
    checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
    stage1_config = {k: v for k, v in checkpoint["model_config"].items() if k != "pooling"}
    model_config = DiMPConfig.from_dict(stage1_config)

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
            json.dumps({"model": model_config.to_dict(), "finetune": serializable}, indent=2) + "\n"
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    accum_steps = max(1, config.grad_accum_steps)
    train_loader, val_loader = build_loaders(config, rank, config.iterations * accum_steps)
    identity_model = DiMPIdentity(model_config)
    identity_model.core.load_state_dict(checkpoint["core"])
    for parameter in identity_model.core.parameters():
        parameter.requires_grad_(False)
    identity_model.to(device)
    head_params = [p for n, p in identity_model.named_parameters() if not n.startswith("core.")]
    model = DDP(identity_model, device_ids=[local_rank]) if world_size > 1 else identity_model
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": config.learning_rate}], weight_decay=config.weight_decay)
    scheduler = cosine_schedule(optimizer, config.warmup_steps, config.iterations)
    use_amp = device.type == "cuda"
    if config.pseudo_gen_aug:
        from dimp.pseudogen import NUM_FAMILIES, apply_pseudo_generator, landmark_parts

        part, parts = landmark_parts()
        part = part.to(device)

    best_auc, no_improve = -1.0, 0
    progress = tqdm(train_loader, total=config.iterations * accum_steps, desc="stage 2", unit="micro-batch",
                    dynamic_ncols=True, miniters=config.log_interval, disable=rank != 0)
    no_sync = model.no_sync if hasattr(model, "no_sync") else nullcontext
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    iteration = 0
    for micro_step, batch in enumerate(progress, start=1):
        model.train()
        identity_model.core.eval()  # frozen encoder
        mesh = batch["mesh"].to(device, non_blocking=True)
        if config.pseudo_gen_aug:
            family = torch.randint(0, NUM_FAMILIES, (mesh.shape[0],), device=device)
            with torch.no_grad():
                mesh = apply_pseudo_generator(
                    mesh, family, part, parts,
                    lambda *shape: torch.rand(*shape, device=device), lambda *shape: torch.randn(*shape, device=device),
                )
        driver_ids = batch["driver_id"].to(device, non_blocking=True)
        is_last_micro_step = micro_step % accum_steps == 0
        with no_sync() if not is_last_micro_step else nullcontext():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = supervised_contrastive_loss(model(mesh), driver_ids, config.temperature)
            (loss / accum_steps).backward()
        accumulated_loss += loss.item()
        if not is_last_micro_step:
            continue
        iteration += 1
        mean_loss, accumulated_loss = accumulated_loss / accum_steps, 0.0
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0 and (iteration == 1 or iteration % config.log_interval == 0):
            LOGGER.info("iteration %d/%d | loss %.5f | lr %.3g", iteration, config.iterations, mean_loss, scheduler.get_last_lr()[0])

        stop_signal = torch.zeros(1, dtype=torch.uint8, device=device)
        if iteration % config.validation_interval == 0 or iteration == config.iterations:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                identity_model.eval()
                metrics = evaluate(identity_model, val_loader, device)
                text = " | ".join(f"{key}={value:.5f}" for key, value in metrics.items())
                LOGGER.info("validation | %s", text)
                tqdm.write(f"validation at {iteration} | {text}")
                if metrics["mean_auc"] > best_auc:
                    best_auc, no_improve = metrics["mean_auc"], 0
                    save_checkpoint(config.output_dir / "best_model.pt", {
                        "iteration": iteration, "model": identity_model.state_dict(),
                        "model_config": model_config.to_dict(), "finetune_config": serializable,
                        "metrics": metrics, "external_pretrained": False,
                    })
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
