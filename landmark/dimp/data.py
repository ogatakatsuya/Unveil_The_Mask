from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def prepare_mesh(landmarks: torch.Tensor, frames: int, *, training: bool, jitter: float = 0.005) -> torch.Tensor:
    """(T, 106, 2) landmarks -> (frames, 106, 2) appearance-normalized displacements.

    Per-frame centroid removal, one scale per sample, removal of each landmark's
    temporal mean position (the avatar's face shape), and normalization to unit
    mean norm. Short samples are padded with their last frame.
    """
    landmarks = torch.as_tensor(landmarks, dtype=torch.float32)
    if landmarks.ndim != 3 or landmarks.shape[-2:] != (106, 2):
        raise ValueError(f"expected (frames, 106, 2), got {landmarks.shape}")
    if landmarks.shape[0] < frames:
        padding = landmarks[-1:].expand(frames - landmarks.shape[0], -1, -1)
        landmarks = torch.cat([landmarks, padding], dim=0)
    elif landmarks.shape[0] > frames:
        if training:
            start = int(torch.randint(landmarks.shape[0] - frames + 1, ()).item())
        else:
            start = (landmarks.shape[0] - frames) // 2
        landmarks = landmarks[start : start + frames]

    centered = landmarks - landmarks.mean(dim=1, keepdim=True)
    normalized = centered / centered.norm(dim=-1).mean().clamp_min(1e-6)
    deviation = normalized - normalized.mean(dim=0, keepdim=True)
    deviation = deviation / deviation.norm(dim=-1).mean().clamp_min(1e-6)
    if training and jitter > 0:
        deviation = deviation + torch.randn_like(deviation) * jitter
    return deviation


class LandmarkDataset(Dataset):
    """Yields ``mesh`` ``(frames, 106, 2)`` and the labels of a sample (when available)."""

    def __init__(
        self,
        root: Path,
        meta_path: Path,
        frames: int,
        *,
        training: bool,
        driver_split: Path | None = None,
        source_split: Path | None = None,
    ) -> None:
        self.root = root
        self.frames = frames
        self.training = training
        metadata = json.loads(meta_path.read_text())
        if driver_split is not None:
            drivers = set(json.loads(driver_split.read_text()))
            metadata = [item for item in metadata if item["driver_id"] in drivers]
        if source_split is not None:
            sources = set(json.loads(source_split.read_text()))
            metadata = [item for item in metadata if item["source_id"] in sources]
        self.meta = [dict(item) for item in metadata if (self.root / item["relpath"]).is_file()]
        if not self.meta:
            raise RuntimeError(f"no samples found for {meta_path}")

        driver_names = sorted({item.get("driver_id", "") for item in self.meta})
        source_names = sorted({item.get("source_id", "") for item in self.meta})
        self.driver_map = {name: index for index, name in enumerate(driver_names)}
        self.source_map = {name: index for index, name in enumerate(source_names)}
        for item in self.meta:
            item["driver_index"] = self.driver_map.get(item.get("driver_id", ""), -1)
            item["source_index"] = self.source_map.get(item.get("source_id", ""), -1)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict:
        item = self.meta[index]
        landmark = blosc2.load_tensor(str(self.root / item["relpath"]))
        return {
            "mesh": prepare_mesh(landmark, self.frames, training=self.training),
            "driver_id": item["driver_index"],
            "source_id": item["source_index"],
            "mode": item.get("mode", ""),
            "set": item.get("set", ""),
            "file": item["relpath"],
        }


class RandomBatchSampler(Sampler[list[int]]):
    """Uniform batches for Stage 1."""

    def __init__(self, dataset_size: int, batch_size: int, iterations: int, seed: int, rank: int = 0) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.iterations = iterations
        self.rng = np.random.default_rng(seed + rank)

    def __len__(self) -> int:
        return self.iterations

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.iterations):
            yield self.rng.choice(self.dataset_size, self.batch_size, replace=False).tolist()


class CrossIdentityBatchSampler(Sampler[list[int]]):
    """Batches that cross drivers with sources: ``drivers_per_batch`` drivers, and
    ``sources_per_driver`` sources recorded with all of them, one sample per pair."""

    def __init__(
        self,
        dataset: LandmarkDataset,
        drivers_per_batch: int,
        sources_per_driver: int,
        iterations: int,
        seed: int,
        rank: int = 0,
    ) -> None:
        self.iterations = iterations
        self.drivers_per_batch = drivers_per_batch
        self.sources_per_driver = sources_per_driver
        self.rng = np.random.default_rng(seed + rank)
        self.by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.meta):
            self.by_pair[item["driver_index"], item["source_index"]].append(index)
        self.drivers = sorted({driver for driver, _ in self.by_pair})
        self.sources = sorted({source for _, source in self.by_pair})
        if drivers_per_batch > len(self.drivers):
            raise ValueError("drivers_per_batch exceeds available driver identities")

    def __len__(self) -> int:
        return self.iterations

    def _sample(self, values: Sequence[int], count: int) -> list[int]:
        return self.rng.choice(values, count, replace=len(values) < count).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.iterations):
            for _attempt in range(100):
                drivers = self._sample(self.drivers, self.drivers_per_batch)
                common_sources = [
                    source for source in self.sources
                    if all((driver, source) in self.by_pair for driver in drivers)
                ]
                if len(common_sources) >= self.sources_per_driver:
                    break
            else:
                raise RuntimeError("could not find enough common sources for a batch")
            sources = self._sample(common_sources, self.sources_per_driver)
            yield [int(self.rng.choice(self.by_pair[driver, source])) for driver in drivers for source in sources]
