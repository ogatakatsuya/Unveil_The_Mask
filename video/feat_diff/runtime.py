from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .objective import binary_auc


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    """Same mode-pair AUC protocol as the other WBA2026 experiments in this repo,
    with mean_auc aligned to the official leaderboard definition: mean of Speaking,
    Non-Speaking, and Cross-Condition AUC, where Cross-Condition pools both
    cross-mode probe/gallery directions into a single AUC.
    """
    model.eval()
    buckets: dict[tuple[str, str], dict[str, list[torch.Tensor]]] = {
        (subset, mode): {"embedding": [], "driver": [], "source": []}
        for subset in ("probe", "gallery")
        for mode in ("speaking", "listening")
    }
    for batch in loader:
        embedding = model(batch["frames"].to(device, non_blocking=True))
        for index, (subset, mode) in enumerate(zip(batch["set"], batch["mode"])):
            key = (subset, mode)
            if key in buckets:
                buckets[key]["embedding"].append(embedding[index].cpu())
                buckets[key]["driver"].append(batch["driver_id"][index].cpu())
                buckets[key]["source"].append(batch["source_id"][index].cpu())

    output = {}
    pair_scores: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for probe_mode in ("speaking", "listening"):
        for gallery_mode in ("speaking", "listening"):
            probe = buckets[("probe", probe_mode)]
            gallery = buckets[("gallery", gallery_mode)]
            if not probe["embedding"] or not gallery["embedding"]:
                continue
            probe_embedding = torch.stack(probe["embedding"])
            gallery_embedding = torch.stack(gallery["embedding"])
            probe_driver = torch.stack(probe["driver"])
            gallery_driver = torch.stack(gallery["driver"])
            scores = probe_embedding @ gallery_embedding.T
            labels = probe_driver[:, None].eq(gallery_driver[None])
            pair_scores[(probe_mode, gallery_mode)] = (scores.flatten(), labels.flatten())
            name = f"{probe_mode}_probe_vs_{gallery_mode}_gallery"
            output[name] = binary_auc(scores.flatten(), labels.flatten())

    leaderboard_terms = []
    if ("speaking", "speaking") in pair_scores:
        leaderboard_terms.append(output["speaking_probe_vs_speaking_gallery"])
    if ("listening", "listening") in pair_scores:
        leaderboard_terms.append(output["listening_probe_vs_listening_gallery"])
    cross_pairs = [
        pair_scores[key]
        for key in (("speaking", "listening"), ("listening", "speaking"))
        if key in pair_scores
    ]
    if cross_pairs:
        cross_scores = torch.cat([scores for scores, _ in cross_pairs])
        cross_labels = torch.cat([labels for _, labels in cross_pairs])
        output["cross_condition_auc"] = binary_auc(cross_scores, cross_labels)
        leaderboard_terms.append(output["cross_condition_auc"])
    output["mean_auc"] = sum(leaderboard_terms) / len(leaderboard_terms)

    # Diagnostic (not part of the official metric): same-avatar vs
    # different-avatar AUC, pooled across all mode pairs. Local proxy for
    # CodaBench's "same generator" / "different generator" split -- the axis
    # landmark-based models here have historically collapsed to chance on.
    # Caveat: our local "different avatar" split only holds out a subset of
    # source_ids from the same overall training distribution, not a fully
    # disjoint pool like CodaBench's held-out avatar templates, so treat
    # these as a relative signal across checkpoints, not an absolute one.
    all_probe_embedding, all_probe_driver, all_probe_source = [], [], []
    all_gallery_embedding, all_gallery_driver, all_gallery_source = [], [], []
    for mode in ("speaking", "listening"):
        all_probe_embedding += buckets[("probe", mode)]["embedding"]
        all_probe_driver += buckets[("probe", mode)]["driver"]
        all_probe_source += buckets[("probe", mode)]["source"]
        all_gallery_embedding += buckets[("gallery", mode)]["embedding"]
        all_gallery_driver += buckets[("gallery", mode)]["driver"]
        all_gallery_source += buckets[("gallery", mode)]["source"]
    if all_probe_embedding and all_gallery_embedding:
        probe_embedding = torch.stack(all_probe_embedding)
        gallery_embedding = torch.stack(all_gallery_embedding)
        probe_driver = torch.stack(all_probe_driver)
        gallery_driver = torch.stack(all_gallery_driver)
        probe_source = torch.stack(all_probe_source)
        gallery_source = torch.stack(all_gallery_source)
        scores = (probe_embedding @ gallery_embedding.T).flatten()
        driver_match = probe_driver[:, None].eq(gallery_driver[None]).flatten()
        source_match = probe_source[:, None].eq(gallery_source[None]).flatten()
        if source_match.any():
            output["same_generator_auc"] = binary_auc(
                scores[source_match], driver_match[source_match]
            )
        if (~source_match).any():
            output["different_generator_auc"] = binary_auc(
                scores[~source_match], driver_match[~source_match]
            )
    return output


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    iteration: int,
    model_config: dict,
    train_config: dict,
    metrics: dict[str, float],
) -> None:
    module = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config,
            "train_config": train_config,
            "metrics": metrics,
            "pretrained": False,
        },
        path,
    )
