from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Callable

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
def evaluate(
    embed: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Verification AUCs on the validation split.

    ``embed`` maps a mesh batch to L2-normalized embeddings. ``mean_auc`` is the
    mean of the speaking, listening and pooled cross-mode AUCs.
    ``same_generator_auc`` / ``different_generator_auc`` split the pairs by
    whether probe and gallery share the source.
    """
    buckets: dict[tuple[str, str], dict[str, list[torch.Tensor]]] = {
        (subset, mode): {"embedding": [], "driver": [], "source": []}
        for subset in ("probe", "gallery")
        for mode in ("speaking", "listening")
    }
    for batch in loader:
        embedding = embed(batch["mesh"].to(device, non_blocking=True))
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
            scores = probe_embedding @ gallery_embedding.T
            labels = torch.stack(probe["driver"])[:, None].eq(
                torch.stack(gallery["driver"])[None]
            )
            pair_scores[(probe_mode, gallery_mode)] = (scores.flatten(), labels.flatten())
            name = f"{probe_mode}_probe_vs_{gallery_mode}_gallery"
            output[name] = binary_auc(scores.flatten(), labels.flatten())

    leaderboard_terms = []
    for mode in ("speaking", "listening"):
        if (mode, mode) in pair_scores:
            leaderboard_terms.append(output[f"{mode}_probe_vs_{mode}_gallery"])
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

    probe_embedding, probe_driver, probe_source = [], [], []
    gallery_embedding, gallery_driver, gallery_source = [], [], []
    for mode in ("speaking", "listening"):
        probe_embedding += buckets[("probe", mode)]["embedding"]
        probe_driver += buckets[("probe", mode)]["driver"]
        probe_source += buckets[("probe", mode)]["source"]
        gallery_embedding += buckets[("gallery", mode)]["embedding"]
        gallery_driver += buckets[("gallery", mode)]["driver"]
        gallery_source += buckets[("gallery", mode)]["source"]
    if probe_embedding and gallery_embedding:
        scores = (torch.stack(probe_embedding) @ torch.stack(gallery_embedding).T).flatten()
        driver_match = torch.stack(probe_driver)[:, None].eq(
            torch.stack(gallery_driver)[None]
        ).flatten()
        source_match = torch.stack(probe_source)[:, None].eq(
            torch.stack(gallery_source)[None]
        ).flatten()
        if source_match.any():
            output["same_generator_auc"] = binary_auc(
                scores[source_match], driver_match[source_match]
            )
        if (~source_match).any():
            output["different_generator_auc"] = binary_auc(
                scores[~source_match], driver_match[~source_match]
            )
    return output


@torch.no_grad()
def evaluate_pretext(pretrainer, loader, device: torch.device, seed: int = 1234) -> dict[str, float]:
    """Pretext losses on the validation samples with a fixed mask/timestep/noise stream."""
    totals: dict[str, float] = {}
    count = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for batch in loader:
        mesh = batch["mesh"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = pretrainer(mesh, generator=generator)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(mesh)
        count += len(mesh)
    return {f"pretext_{key}": value / count for key, value in totals.items()}


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
