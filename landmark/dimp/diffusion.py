from __future__ import annotations

import math

import torch
from torch import nn


class CosineSchedule:
    """DDPM cosine noise schedule (Nichol & Dhariwal), timesteps ``t in [1, T]``.

    ``alpha_bar(t)`` is the cumulative product at 1-indexed step ``t`` so that
    ``x_t = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) eps``.
    """

    def __init__(self, num_steps: int = 1000, offset: float = 0.008) -> None:
        self.num_steps = num_steps
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        f = torch.cos((steps / num_steps + offset) / (1 + offset) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        self._alpha_bar = torch.cumprod(1 - betas, dim=0).float()  # index t-1

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self._alpha_bar.to(t.device)[t - 1]

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """``t``: ``(B,)`` long; ``x0``/``noise``: ``(B, ...)``."""
        alpha_bar = self.alpha_bar(t).view(-1, *([1] * (x0.dim() - 1)))
        return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    def stratified_intervals(self, count: int) -> list[tuple[int, int]]:
        """Eq. (6): ``d = floor(T / h)``; ``Q_i = [d i + 1, d (i + 1)]``, last ``[d (h-1) + 1, T]``."""
        d = self.num_steps // count
        intervals = []
        for i in range(count):
            low = d * i + 1
            high = self.num_steps if i == count - 1 else d * (i + 1)
            intervals.append((low, high))
        return intervals


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        half = dim // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * half, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float()[:, None] * self.frequencies[None]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))
