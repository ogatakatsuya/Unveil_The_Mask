from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    driver_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE: same driver is positive; every distinct driver is negative."""
    if embeddings.ndim != 2 or driver_ids.ndim != 1:
        raise ValueError("embeddings must be 2D and driver_ids must be 1D")
    if embeddings.shape[0] != driver_ids.shape[0]:
        raise ValueError("batch dimensions do not match")

    logits = embeddings.float() @ embeddings.float().T / temperature
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = driver_ids[:, None].eq(driver_ids[None]) & ~self_mask
    if not positive_mask.all(dim=1).any() and not positive_mask.any():
        raise ValueError("each batch must contain at least two samples per driver")
    if not positive_mask.any(dim=1).all():
        raise ValueError("every sample must have a positive pair")

    logits = logits.masked_fill(self_mask, -torch.inf)
    log_prob = F.log_softmax(logits, dim=1)
    positive_count = positive_mask.sum(dim=1)
    return -(
        log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count
    ).mean()


@torch.no_grad()
def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUC with average ranks for ties, without a sklearn dependency."""
    scores = scores.float().cpu()
    labels = labels.bool().cpu()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    rank_sum = ranks[labels].sum().item()
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
