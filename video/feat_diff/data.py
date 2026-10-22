from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# DataLoader workers each decode a video; per-worker thread pools only oversubscribe the CPU.
cv2.setNumThreads(1)


FACE_BOX_DIR = Path(__file__).resolve().parents[1] / "cache"


def load_face_extents(split: str) -> dict[str, np.ndarray]:
    """mp4 relpath -> (x_min, y_min, x_max, y_max) of the clip's landmarks over
    all frames, from ``scripts/compute_face_boxes.py`` (``split``: train /
    test-gen1 / test-gen2)."""
    path = FACE_BOX_DIR / f"face_boxes-{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run scripts/compute_face_boxes.py first")
    data = np.load(path)
    return dict(zip(data["relpath"].tolist(), data["extent"]))


def square_face_box(
    extent: np.ndarray, margin: float, jitter: float = 0.0, rng: np.random.Generator | None = None
) -> tuple[int, int, int]:
    """Square crop ``(x0, y0, side)`` around a clip's landmark extent: side =
    ``margin`` x the longer edge. With ``jitter`` > 0 (training only), the side
    is scaled by U(1-jitter, 1+jitter) and the centre shifted by
    U(-jitter, jitter) x side on each axis, one draw per clip, so the model
    cannot rely on the face sitting at an exact position/scale."""
    x_min, y_min, x_max, y_max = (float(v) for v in extent)
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = max(x_max - x_min, y_max - y_min) * margin
    if jitter > 0:
        rng = rng or np.random.default_rng()
        side *= rng.uniform(1 - jitter, 1 + jitter)
        center_x += rng.uniform(-jitter, jitter) * side
        center_y += rng.uniform(-jitter, jitter) * side
    side_px = max(int(round(side)), 8)
    return int(round(center_x - side_px / 2)), int(round(center_y - side_px / 2)), side_px


def crop_square(gray: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """Crop ``box`` from a frame; parts outside the frame are filled with 0 (black)."""
    x0, y0, side = box
    height, width = gray.shape
    out = np.zeros((side, side), dtype=gray.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, width), min(y0 + side, height)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    return out


def load_grayscale_clip(
    path: Path, frame_size: int, box: tuple[int, int, int] | None = None
) -> torch.Tensor:
    """Decode every frame as grayscale, optionally crop the same square ``box``
    from each (face crop), and resize to ``frame_size``."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if box is not None:
                gray = crop_square(gray, box)
            resized = cv2.resize(gray, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            frames.append(resized)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


_GAUSSIAN_BLUR_KERNELS: dict[int, torch.Tensor] = {}


def prepare_clip(
    path: Path,
    clip_length: int,
    frame_size: int,
    *,
    training: bool,
    box: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    frames = load_grayscale_clip(path, frame_size, box)  # (N, H, W)
    total = frames.shape[0]
    if total < clip_length:
        padding = frames[-1:].expand(clip_length - total, -1, -1)
        frames = torch.cat([frames, padding], dim=0)
    elif total > clip_length:
        if training:
            start = int(torch.randint(total - clip_length + 1, ()).item())
        else:
            start = (total - clip_length) // 2
        frames = frames[start : start + clip_length]
    # Per-clip z-score normalization removes brightness/contrast offsets. Static
    # avatar appearance itself is cancelled architecturally by the model's
    # inter-frame differencing, not by this preprocessing step.
    frames = (frames - frames.mean()) / frames.std().clamp_min(1e-6)
    return frames.unsqueeze(1)  # (T, 1, H, W)


class VideoClipDataset(Dataset):
    def __init__(
        self,
        root: Path,
        meta_path: Path,
        clip_length: int,
        frame_size: int,
        *,
        training: bool,
        driver_split: Path | None = None,
        source_split: Path | None = None,
        face_extents: dict[str, np.ndarray] | None = None,
        crop_margin: float = 1.25,
        crop_jitter: float = 0.0,
    ) -> None:
        """``face_extents`` (from ``load_face_extents``) turns on the face crop:
        each clip is cropped to a square around its landmarks (``crop_margin``
        x the longer edge; ``crop_jitter`` applies only when ``training``)."""
        self.root = root
        self.face_extents = face_extents
        self.crop_margin = crop_margin
        self.crop_jitter = crop_jitter
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.training = training
        metadata = json.loads(meta_path.read_text())
        if driver_split is not None:
            drivers = set(json.loads(driver_split.read_text()))
            metadata = [item for item in metadata if item["driver_id"] in drivers]
        if source_split is not None:
            sources = set(json.loads(source_split.read_text()))
            metadata = [item for item in metadata if item["source_id"] in sources]
        self.meta = [
            dict(item) for item in metadata if (self.root / item["relpath"]).is_file()
        ]
        if not self.meta:
            raise RuntimeError(f"no samples found for {meta_path}")
        if face_extents is not None:
            missing = [item["relpath"] for item in self.meta if item["relpath"] not in face_extents]
            if missing:
                raise KeyError(f"{len(missing)} clips have no face box, e.g. {missing[0]}")

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
        box = None
        if self.face_extents is not None:
            box = square_face_box(
                self.face_extents[item["relpath"]],
                self.crop_margin,
                self.crop_jitter if self.training else 0.0,
                # torch's RNG, so the DataLoader's per-worker seeding applies
                np.random.default_rng(int(torch.randint(2**31 - 1, ()).item())),
            )
        clip = prepare_clip(
            self.root / item["relpath"],
            self.clip_length,
            self.frame_size,
            training=self.training,
            box=box,
        )
        return {
            "frames": clip,
            "driver_id": item["driver_index"],
            "source_id": item["source_index"],
            "mode": item.get("mode", ""),
            "set": item.get("set", ""),
            "file": item["relpath"],
        }


class CrossIdentityBatchSampler(Sampler[list[int]]):
    """Batches that cross drivers with sources: every positive pair shows the same
    driver on different avatars, and every sample has same-avatar negatives."""

    def __init__(
        self,
        dataset: VideoClipDataset,
        drivers_per_batch: int,
        sources_per_driver: int,
        iterations: int,
        seed: int,
        rank: int = 0,
    ) -> None:
        self.iterations = iterations
        self.dataset = dataset
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
                    source
                    for source in self.sources
                    if all((driver, source) in self.by_pair for driver in drivers)
                ]
                if len(common_sources) >= self.sources_per_driver:
                    break
            else:
                raise RuntimeError("could not find enough common sources for a batch")
            sources = self._sample(common_sources, self.sources_per_driver)
            batch = []
            for driver in drivers:
                for source in sources:
                    candidates = self.by_pair[driver, source]
                    batch.append(int(self.rng.choice(candidates)))
            yield batch
