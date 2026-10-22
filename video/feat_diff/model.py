from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    frame_size: int = 128
    clip_length: int = 100
    embedding_dim: int = 256
    neighbor_k: int = 4
    meta_kernel_size: int = 32
    fcc_dim: int = 64
    # Landmark-guided face crop (scripts/compute_face_boxes.py): a square box of
    # crop_margin x the longer edge of the sample's landmark bounding box.
    face_crop: bool = False
    crop_margin: float = 1.25
    # "feat": F(I[t+1]) - F(I[t]) (feature differencing);
    # "pixel": F(I[t+1] - I[t]) (pixel differencing).
    diff_mode: str = "feat"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        """Builds the config from a checkpoint of the research code, which also
        stored a frame lag (always 1 for the released models)."""
        if values.get("diff_lag", 1) != 1 or values.get("diff_mode", "feat") not in ("feat", "pixel"):
            raise ValueError(f"checkpoint uses settings not supported by this release: {values}")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


class ConvStack(nn.Module):
    """Grayscale frame (or stacked difference channels) -> (fcc_dim*2, H/8, W/8) feature map."""

    def __init__(self, out_dim: int, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 4, stride=2), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 3, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 2, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCC(nn.Module):
    """Factorized cross-correlation block (micro-expression-aware, à la F5C/MOL)."""

    def __init__(self, dim: int, meta_kernel_size: int, use_pe: bool = True, bias: bool = True) -> None:
        super().__init__()
        self.dim, self.use_pe = dim, use_pe
        self.pre_norm_1 = nn.BatchNorm2d(dim)
        self.pre_norm_2 = nn.BatchNorm2d(dim)
        self.mk_1_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.mk_1_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight
        self.mk_2_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.mk_2_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight
        if bias:
            self.b1H = nn.Parameter(torch.randn(dim))
            self.b1W = nn.Parameter(torch.randn(dim))
            self.b2H = nn.Parameter(torch.randn(dim))
            self.b2W = nn.Parameter(torch.randn(dim))
        else:
            self.b1H = self.b1W = self.b2H = self.b2W = None
        if use_pe:
            self.pe_1H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.pe_1W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.pe_2H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.pe_2W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))

    def _kernels(self, size: int):
        return (
            self.mk_1_H[:, :, :size, :], self.mk_1_W[:, :, :, :size],
            self.mk_2_H[:, :, :size, :], self.mk_2_W[:, :, :, :size],
        )

    def _positions(self, size: int):
        return (
            self.pe_1H[:, :, :size, :].expand(1, self.dim, size, size),
            self.pe_1W[:, :, :, :size].expand(1, self.dim, size, size),
            self.pe_2H[:, :, :size, :].expand(1, self.dim, size, size),
            self.pe_2W[:, :, :, :size].expand(1, self.dim, size, size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1r, x2r = x1, x2
        _, _, size, _ = x1.shape
        k1h, k1w, k2h, k2w = self._kernels(size)
        if self.use_pe:
            p1h, p1w, p2h, p2w = self._positions(size)
            x1, x2 = x1 + p1h, x2 + p1w
        x1 = self.pre_norm_1(x1)
        x2 = self.pre_norm_2(x2)
        x1 = F.conv2d(torch.cat((x1, x1[:, :, :-1, :]), 2), weight=k1h, bias=self.b1H, groups=self.dim)
        x2 = F.conv2d(torch.cat((x2, x2[:, :, :, :-1]), 3), weight=k1w, bias=self.b1W, groups=self.dim)
        if self.use_pe:
            x1, x2 = x1 + p2w, x2 + p2h
        x1 = F.conv2d(torch.cat((x1, x1[:, :, :, :-1]), 3), weight=k2w, bias=self.b2W, groups=self.dim)
        x2 = F.conv2d(torch.cat((x2, x2[:, :, :-1, :]), 2), weight=k2h, bias=self.b2H, groups=self.dim)
        return torch.cat((x1r + x1, x2r + x2), dim=1)


def graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-NN edge features over spatial positions (cosine similarity), DGCNN-style."""
    batch, channels, positions = x.size()
    x_normalized = F.normalize(x, p=2, dim=1)
    similarity = torch.bmm(x_normalized.transpose(1, 2), x_normalized)
    neighbor_index = similarity.topk(k=k, dim=-1)[1]
    base = torch.arange(batch, device=x.device).view(-1, 1, 1) * positions
    neighbor_index = (neighbor_index + base).view(-1)
    flat = x.transpose(1, 2).contiguous().view(batch * positions, -1)
    neighbor_feature = flat[neighbor_index].view(batch, positions, k, channels)
    center = x.transpose(1, 2).view(batch, positions, 1, channels).expand_as(neighbor_feature)
    return (neighbor_feature - center).permute(0, 3, 1, 2)


class CCC(nn.Module):
    """Cross-correlation-conv edge aggregation: center feature + max over k-NN edges."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, center: torch.Tensor, neighbor_edges: torch.Tensor) -> torch.Tensor:
        return F.relu(center + neighbor_edges.max(dim=-1)[0], inplace=False)


class TemporalIdentityHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 64, (3, 3, 3), stride=(1, 2, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, (3, 3, 3), stride=(1, 2, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, embedding_dim),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        hidden = self.net(volume).squeeze(-1).flatten(1)
        return F.normalize(self.fc(hidden), p=2, dim=-1)


class FeatDiffModel(nn.Module):
    """Micro-expression-aware avatar fingerprinting via inter-frame feature differencing.

    Port of Chapariniya et al., "Micro-Expression-Aware Avatar Fingerprinting via
    Inter-Frame Feature Differencing" (IEEE FG 2026): each frame is independently
    encoded with a shared F5C-style backbone (ConvStack -> FCC -> k-NN edge conv),
    then consecutive frame features are subtracted so temporally stable appearance
    (avatar identity/shape) cancels by construction while driver-specific
    micro-motion survives into the temporal identity head.
    https://github.com/Mchapariniya/TrustFA-FG2026
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.diff_mode not in ("feat", "pixel"):
            raise ValueError(f"unknown diff_mode {config.diff_mode!r}")
        self.conv_stack = ConvStack(out_dim=config.fcc_dim * 2)
        self.fcc = FCC(dim=config.fcc_dim, meta_kernel_size=config.meta_kernel_size)
        self.ccc = CCC()
        self.head = TemporalIdentityHead(
            in_channels=config.fcc_dim * 2, embedding_dim=config.embedding_dim
        )

    def _frame_feature(self, frame: torch.Tensor) -> torch.Tensor:
        feature = self.conv_stack(frame)
        feature = self.fcc(feature)
        batch, channels, height, width = feature.shape
        flat = feature.reshape(batch, channels, height * width)
        edges = graph_feature(flat, k=self.config.neighbor_k)
        aggregated = self.ccc(flat, edges)
        return (flat + aggregated).reshape(batch, channels, height, width)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (batch, time, 1, H, W) grayscale -> (batch, embedding_dim)."""
        if frames.ndim != 5 or frames.shape[2] != 1:
            raise ValueError(f"expected (batch, time, 1, H, W), got {frames.shape}")
        time = frames.shape[1]
        if time < 2:
            raise ValueError("need at least two frames to compute frame differences")
        if self.config.diff_mode == "pixel":
            diffs = [self._frame_feature(frames[:, t + 1] - frames[:, t]) for t in range(time - 1)]
        else:
            features = [self._frame_feature(frames[:, t]) for t in range(time)]
            diffs = [features[t + 1] - features[t] for t in range(time - 1)]
        volume = torch.stack(diffs, dim=-1)  # (batch, channels, H', W', time-1)
        return self.head(volume)
