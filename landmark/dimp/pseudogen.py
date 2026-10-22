"""Pseudo-generator transforms (pseudo-generator randomization of Stage 1).

An unseen avatar generator is modelled as an unknown *transfer function* on
the driver's motion. Each training sample gets one family:

  0 clean
  1 global smooth time warp   (speed factor U[0.85, 1.15] + a slow sinusoidal warp)
  2 regional lag              (each facial part delayed / advanced by up to 3 frames)
  3 regional frequency resp.  (each part low-passed or high-frequency-boosted)
  4 correlated tracking noise (temporally smooth jitter, independent per landmark)
  5 composite                 (two of 1-4 applied in sequence)

All transforms act on the normalized deviation mesh (B, T, V, 2) produced by
prepare_mesh and are followed by the same unit-mean-norm normalization, so
static shape and global amplitude stay removed. Only the dataset's own data
are used; nothing is learned here.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NUM_FAMILIES = 6


def _renorm(mesh: torch.Tensor) -> torch.Tensor:
    mesh = mesh - mesh.mean(dim=1, keepdim=True)
    return mesh / mesh.norm(dim=-1).mean(dim=(1, 2), keepdim=True)[..., None].clamp_min(1e-6)


def _time_warp(x: torch.Tensor, rand) -> torch.Tensor:
    b, t = x.shape[:2]
    c = (t - 1) / 2
    grid = torch.arange(t, device=x.device, dtype=x.dtype)[None]
    rate = 0.85 + 0.3 * rand(b, 1)
    amp, freq, phase = 2.0 * rand(b, 1), 0.5 + 1.5 * rand(b, 1), 2 * math.pi * rand(b, 1)
    pos = (c + rate * (grid - c) + amp * torch.sin(2 * math.pi * freq * grid / t + phase)).clamp(0, t - 1)
    lo = pos.floor().long().clamp(max=t - 1)
    hi = (lo + 1).clamp(max=t - 1)
    w = (pos - lo.to(pos.dtype))[..., None, None]
    index = lambda i: i[..., None, None].expand(-1, -1, *x.shape[2:])  # noqa: E731
    return x.gather(1, index(lo)) * (1 - w) + x.gather(1, index(hi)) * w


def _regional_lag(x: torch.Tensor, part: torch.Tensor, parts: int, rand) -> torch.Tensor:
    b, t = x.shape[:2]
    lag = (rand(b, parts) * 7).long() - 3  # [-3, 3] frames per part
    per_vertex = lag[:, part]  # (B, V)
    src = (torch.arange(t, device=x.device)[None, :, None] - per_vertex[:, None, :]).clamp(0, t - 1)  # (B, T, V)
    return x.gather(1, src[..., None].expand(-1, -1, -1, x.shape[-1]))


def _smooth(x: torch.Tensor, width: int) -> torch.Tensor:
    """Moving average over time (edge-padded), x (B, T, V, C)."""
    b, t, v, c = x.shape
    y = x.permute(0, 2, 3, 1).reshape(b * v * c, 1, t)
    y = F.pad(y, (width // 2, width // 2), mode="replicate")
    y = F.avg_pool1d(y, width, stride=1)
    return y.reshape(b, v, c, t).permute(0, 3, 1, 2)


def _frequency_response(x: torch.Tensor, part: torch.Tensor, parts: int, rand) -> torch.Tensor:
    b = x.shape[0]
    low = _smooth(x, 5)
    # per part: alpha in [-0.6, 0.6]; <0 blends towards the low-pass, >0 boosts the high frequencies
    alpha = (1.2 * rand(b, parts) - 0.6)[:, part][:, None, :, None]
    return x + alpha * (x - low)


def _tracking_noise(x: torch.Tensor, rand, randn) -> torch.Tensor:
    b = x.shape[0]
    noise = _smooth(randn(*x.shape), 7)
    scale = 0.03 + 0.07 * rand(b, 1, 1, 1)  # relative to the unit mean norm of the mesh
    return x + scale * noise / noise.std().clamp_min(1e-6)


def apply_pseudo_generator(
    mesh: torch.Tensor, family: torch.Tensor, part: torch.Tensor, parts: int, rand, randn
) -> torch.Tensor:
    """mesh (B, T, V, 2), family (B,) in [0, NUM_FAMILIES), part (V,) part id of each landmark."""
    x = mesh.float()
    out = x.clone()
    single = {
        1: lambda y: _time_warp(y, rand),
        2: lambda y: _regional_lag(y, part, parts, rand),
        3: lambda y: _frequency_response(y, part, parts, rand),
        4: lambda y: _tracking_noise(y, rand, randn),
    }
    for fam, fn in single.items():
        sel = family == fam
        if sel.any():
            out[sel] = fn(x[sel])
    sel = family == 5
    if sel.any():
        y = x[sel]
        first = (rand(int(sel.sum())) * 4).long() + 1
        second = (first - 1 + 1 + (rand(int(sel.sum())) * 3).long()) % 4 + 1  # a different family
        for order in (first, second):
            for fam, fn in single.items():
                pick = order == fam
                if pick.any():
                    y[pick] = fn(y[pick])
        out[sel] = y
    changed = family > 0
    out[changed] = _renorm(out[changed])
    return out.to(mesh.dtype)


def landmark_parts() -> tuple[torch.Tensor, int]:
    """Part id of each of the 106 landmarks (pupils merged into the eyes, nose bridge into the nose) and the part count."""
    import json

    from . import regions

    names = json.loads(regions.REGIONS_PATH.read_text())
    merge = {"left_pupil": "left_eye", "right_pupil": "right_eye", "nose_bridge": "nose_tip"}
    labels = [merge.get(names[str(v)], names[str(v)]) for v in range(106)]
    vocab = sorted(set(labels))
    return torch.tensor([vocab.index(n) for n in labels]), len(vocab)
