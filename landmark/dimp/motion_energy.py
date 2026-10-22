"""Per-token motion energy, the target of the ME auxiliary task."""
from __future__ import annotations

import torch


def token_motion_energy(motion: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``motion``: ``(..., K, n, l-1, 2)`` per-token inter-frame displacement
    (already unit-variance-normalised by ``TubeLayout``). ``valid``: ``(K, n)``
    or ``(..., K, n)`` padding mask (batched valid is needed when tokens have
    been gathered/reordered per-sample, e.g. by a random visible/masked
    split -- broadcasts against an unbatched mask too). Returns ``(..., K)``
    = mean squared displacement magnitude per token, i.e. how much that tube
    moved during the clip."""
    energy = motion.pow(2).sum(dim=-1)  # (..., K, n, l-1)
    mask = valid[..., None].expand_as(energy).to(energy.dtype)  # broadcast BEFORE summing
    numerator = (energy * mask).sum(dim=(-2, -1))
    denominator = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
    return numerator / denominator
