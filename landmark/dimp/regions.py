"""Hand-annotated facial region of each of the 106 landmarks (``landmark_regions.json``).

The 106-point topology is the same for every sample, so this single annotation
applies to the training and test data alike.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REGIONS_PATH = Path(__file__).resolve().parent / "landmark_regions.json"
NUM_LANDMARKS = 106


def load_region_names(exclude: tuple[str, ...] = ()) -> list[str]:
    """Region names in sorted order (the index is the region id); ``exclude`` drops names."""
    mapping = json.loads(REGIONS_PATH.read_text())
    return sorted(set(mapping.values()) - set(exclude))


def load_region_ids(exclude: tuple[str, ...] = ()) -> torch.Tensor:
    """``(106,)`` long tensor: region id (index into ``load_region_names(exclude)``)
    for each landmark, or ``-1`` for a landmark in an excluded region. Raises
    if any non-excluded index is unlabelled."""
    mapping = json.loads(REGIONS_PATH.read_text())
    names = load_region_names(exclude)
    name_to_id = {name: i for i, name in enumerate(names)}
    missing = [
        i for i in range(NUM_LANDMARKS)
        if str(i) not in mapping and mapping.get(str(i)) not in exclude
    ]
    if missing:
        raise ValueError(f"landmark_regions.json is missing indices {missing}")
    return torch.tensor([name_to_id.get(mapping[str(i)], -1) for i in range(NUM_LANDMARKS)])


def region_motion_energy(mesh: torch.Tensor, region_ids: torch.Tensor, num_regions: int) -> torch.Tensor:
    """``mesh``: ``(B, T, 106, 2)`` deviation channel. ``region_ids``: ``(106,)``
    landmark -> region id (``-1`` = excluded, from ``load_region_ids(exclude=...)``).
    Returns ``(B, num_regions)`` = mean squared frame-to-frame displacement of
    each region's landmarks over the clip -- "how much did the mouth/eyes/
    brows/... move", using real anatomical regions rather than a blind
    spatial partition."""
    velocity = mesh[:, 1:] - mesh[:, :-1]  # (B, T-1, 106, 2)
    energy = velocity.pow(2).sum(dim=-1)  # (B, T-1, 106)
    batch = mesh.shape[0]
    result = mesh.new_zeros(batch, num_regions)
    for region in range(num_regions):
        members = region_ids == region
        result[:, region] = energy[:, :, members].mean(dim=(1, 2))
    return result


def eye_shape_target(mesh: torch.Tensor, region_ids: torch.Tensor, region_names: list[str]) -> torch.Tensor:
    """"Shape and motion" of each eye, not just how much it moved: per frame,
    the eye region's landmark bounding-box aspect ratio (height / width) is a
    coarse but standard proxy for how open the eye is (the same idea as the
    eye-aspect-ratio used for blink detection). Returns ``(B, 4)`` =
    ``(left_mean, left_std, right_mean, right_std)`` of that ratio over the
    clip -- the mean captures the eye's typical openness (shape), the std
    captures how much it changes (blinking/squinting dynamics, motion)."""
    batch = mesh.shape[0]
    values = []
    for eye in ("left_eye", "right_eye"):
        if eye not in region_names:
            values += [mesh.new_zeros(batch), mesh.new_zeros(batch)]
            continue
        members = region_ids == region_names.index(eye)
        points = mesh[:, :, members]  # (B, T, n, 2)
        height = points[..., 1].amax(dim=-1) - points[..., 1].amin(dim=-1)  # (B, T)
        width = (points[..., 0].amax(dim=-1) - points[..., 0].amin(dim=-1)).clamp_min(1e-6)
        ratio = height / width
        values += [ratio.mean(dim=-1), ratio.std(dim=-1)]
    return torch.stack(values, dim=-1)
