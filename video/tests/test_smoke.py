from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from feat_diff.data import prepare_clip
from feat_diff.model import FeatDiffModel, ModelConfig
from feat_diff.objective import cross_source_alignment_loss, supervised_contrastive_loss


def _small_config(**overrides) -> ModelConfig:
    defaults = dict(
        frame_size=32,
        clip_length=6,
        embedding_dim=8,
        neighbor_k=2,
        meta_kernel_size=8,
        fcc_dim=4,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_forward_loss_and_backward() -> None:
    config = _small_config()
    model = FeatDiffModel(config)
    frames = torch.randn(6, config.clip_length, 1, config.frame_size, config.frame_size)
    driver_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    embeddings = model(frames)
    assert embeddings.shape == (6, config.embedding_dim)
    torch.testing.assert_close(
        embeddings.norm(dim=1), torch.ones(6), atol=1e-5, rtol=1e-5
    )
    loss = supervised_contrastive_loss(embeddings, driver_ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.fc[-1].weight.grad is not None


def test_cross_source_alignment_loss() -> None:
    config = _small_config()
    model = FeatDiffModel(config)
    frames = torch.randn(6, config.clip_length, 1, config.frame_size, config.frame_size)
    driver_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    source_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    embeddings = model(frames)
    loss = cross_source_alignment_loss(embeddings, driver_ids, source_ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.fc[-1].weight.grad is not None

    # No cross-source positive pairs at all -> zero loss, no crash.
    same_source = torch.zeros_like(source_ids)
    zero_loss = cross_source_alignment_loss(embeddings.detach(), driver_ids, same_source)
    assert zero_loss.item() == 0.0


def test_appearance_invariance() -> None:
    """Core design claim: two different *static* clips (zero motion, different
    appearance) must produce the identical embedding, since inter-frame
    differencing makes every diff exactly zero regardless of frame content.
    """
    config = _small_config()
    model = FeatDiffModel(config)
    model.eval()
    frame_a = torch.randn(1, 1, config.frame_size, config.frame_size)
    frame_b = torch.randn(1, 1, config.frame_size, config.frame_size)
    static_a = frame_a.unsqueeze(1).expand(
        1, config.clip_length, 1, config.frame_size, config.frame_size
    )
    static_b = frame_b.unsqueeze(1).expand(
        1, config.clip_length, 1, config.frame_size, config.frame_size
    )
    with torch.no_grad():
        embedding_a = model(static_a)
        embedding_b = model(static_b)
    torch.testing.assert_close(embedding_a, embedding_b, atol=1e-6, rtol=1e-6)


def test_prepare_clip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64)
        )
        rng = np.random.default_rng(0)
        for _ in range(10):
            writer.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        writer.release()

        result = prepare_clip(path, clip_length=16, frame_size=32, training=False)
        assert result.shape == (16, 1, 32, 32)
        assert torch.isfinite(result).all()

        result_crop = prepare_clip(path, clip_length=4, frame_size=32, training=False)
        assert result_crop.shape == (4, 1, 32, 32)


def test_square_face_box_and_crop():
    import numpy as np

    from feat_diff.data import crop_square, square_face_box

    extent = np.array([100.0, 150.0, 300.0, 400.0])  # w 200, h 250 -> side 250 * 1.25
    x0, y0, side = square_face_box(extent, 1.25)
    assert side == 312 and (x0 + side / 2, y0 + side / 2) == (200, 275)
    rng = np.random.default_rng(0)
    for _ in range(50):  # jitter stays within +-5% scale and shift
        jx0, jy0, jside = square_face_box(extent, 1.25, 0.05, rng)
        assert 0.95 * 312 - 1 <= jside <= 1.05 * 312 + 1
        assert abs(jx0 + jside / 2 - 200) <= 0.05 * jside + 1

    frame = np.arange(512 * 512, dtype=np.float64).reshape(512, 512) % 251 + 1
    inside = crop_square(frame, (10, 20, 100))
    assert np.array_equal(inside, frame[20:120, 10:110])
    edge = crop_square(frame, (-30, 480, 64))  # partly outside -> zero padding
    assert edge.shape == (64, 64)
    assert (edge[:, :30] == 0).all() and (edge[32:] == 0).all()
    assert np.array_equal(edge[:32, 30:], frame[480:512, 0:34])


def test_feature_and_pixel_differencing():
    torch.manual_seed(0)
    frames = torch.randn(2, 6, 1, 32, 32)
    base = FeatDiffModel(_small_config()).eval()
    pixel = FeatDiffModel(_small_config(diff_mode="pixel")).eval()
    pixel.load_state_dict(base.state_dict())
    with torch.no_grad():
        feats = [base._frame_feature(frames[:, t]) for t in range(6)]
        manual = base.head(torch.stack([feats[t + 1] - feats[t] for t in range(5)], dim=-1))
        assert torch.allclose(base(frames), manual, atol=1e-6)
        diffs = [pixel._frame_feature(frames[:, t + 1] - frames[:, t]) for t in range(5)]
        assert torch.allclose(pixel(frames), pixel.head(torch.stack(diffs, dim=-1)), atol=1e-6)
