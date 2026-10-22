"""Auxiliary behavioral tasks of Stage 1: motion energy (ME), region motion (RM)
and eye shape (ES). Their targets are computed from the input landmarks, so
they need no labels; each task has a small head on the visible-token features.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import regions
from .motion_energy import token_motion_energy


class AuxTaskContext:
    """What the tasks read, built once per forward pass."""

    __slots__ = ("mesh", "pooled", "config", "extra")

    def __init__(self, mesh: torch.Tensor, pooled: torch.Tensor, config, extra: dict) -> None:
        self.mesh = mesh      # (B, T, 106, 2) normalized landmarks (the pretraining input)
        self.pooled = pooled  # (B, dim) mean of the visible-token features
        self.config = config
        self.extra = extra    # z_vis (B, Kv, dim), motion_vis (B, Kv, n, l-1, 2), point_valid_vis (B, Kv, n)


def _init_head(head: nn.Module) -> None:
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)


class MotionEnergyAuxTask(nn.Module):
    """ME: mean squared inter-frame displacement of each visible token, predicted from its own feature."""

    name = "motion_energy"

    def __init__(self, config) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim // 2), nn.GELU(), nn.Linear(config.dim // 2, 1)
        )
        _init_head(self.head)

    def forward(self, ctx: AuxTaskContext) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            target = token_motion_energy(ctx.extra["motion_vis"], ctx.extra["point_valid_vis"])  # (B, Kv)
        pred = self.head(ctx.extra["z_vis"]).float().squeeze(-1)
        return {"loss": F.mse_loss(pred, target.float())}


class RegionMotionAuxTask(nn.Module):
    """RM: mean squared inter-frame displacement of each annotated facial region
    (except the jaw contour) over the sample, predicted from the pooled feature."""

    name = "region_motion"

    def __init__(self, config) -> None:
        super().__init__()
        exclude = tuple(config.region_motion_exclude)
        self.region_ids = regions.load_region_ids(exclude=exclude)
        self.num_regions = len(regions.load_region_names(exclude=exclude))
        hidden = config.embedding_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(config.dim), nn.Linear(config.dim, hidden), nn.GELU(), nn.Linear(hidden, self.num_regions)
        )
        _init_head(self.head)

    def forward(self, ctx: AuxTaskContext) -> dict[str, torch.Tensor]:
        with torch.no_grad(), torch.autocast(device_type=ctx.mesh.device.type, enabled=False):
            target = regions.region_motion_energy(ctx.mesh.float(), self.region_ids.to(ctx.mesh.device), self.num_regions)
        pred = self.head(ctx.pooled).float()
        return {"loss": F.mse_loss(pred, target)}


class EyeShapeAuxTask(nn.Module):
    """ES: mean and standard deviation over the sample of each eye's openness
    (height-to-width ratio of the eye landmarks), predicted from the pooled feature."""

    name = "eye_shape"

    def __init__(self, config) -> None:
        super().__init__()
        self.region_ids = regions.load_region_ids()
        self.region_names = regions.load_region_names()
        self.head = nn.Sequential(
            nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim // 2), nn.GELU(), nn.Linear(config.dim // 2, 4)
        )
        _init_head(self.head)

    def forward(self, ctx: AuxTaskContext) -> dict[str, torch.Tensor]:
        with torch.no_grad(), torch.autocast(device_type=ctx.mesh.device.type, enabled=False):
            target = regions.eye_shape_target(ctx.mesh.float(), self.region_ids.to(ctx.mesh.device), self.region_names)
        pred = self.head(ctx.pooled).float()
        return {"loss": F.mse_loss(pred, target)}


TASK_REGISTRY: dict[str, type[nn.Module]] = {
    MotionEnergyAuxTask.name: MotionEnergyAuxTask,
    RegionMotionAuxTask.name: RegionMotionAuxTask,
    EyeShapeAuxTask.name: EyeShapeAuxTask,
}


def build_aux_tasks(config) -> nn.ModuleDict:
    unknown = set(config.aux_tasks) - set(TASK_REGISTRY)
    if unknown:
        raise ValueError(f"unknown aux task(s) {sorted(unknown)}; available: {sorted(TASK_REGISTRY)}")
    return nn.ModuleDict({name: TASK_REGISTRY[name](config) for name in config.aux_tasks})


def aux_task_weight(config, name: str) -> float:
    return getattr(config, f"{name}_aux_weight")
