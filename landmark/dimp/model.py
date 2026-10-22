"""Diffusion masked pretraining (DiMP) on landmark tokens, and the identity model.

Follows "Diffusion Masked Pretraining for Dynamic Point Cloud" (Zhang et al.),
adapted to spatio-temporal tokens of the 106 facial landmarks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

from .aux_tasks import AuxTaskContext, aux_task_weight, build_aux_tasks
from .diffusion import CosineSchedule, TimeEmbedding
from .tubes import TubeLayout


@dataclass
class DiMPConfig:
    frames: int = 100
    tube_frames: int = 4
    num_groups: int = 16
    group_size: int = 7  # largest group; overwritten from the statistics file
    # encoder (dual-stream Transformer)
    dim: int = 256
    depth: int = 6
    heads: int = 8
    # decoder (discarded after pretraining)
    dec_dim: int = 256
    dec_depth: int = 4
    dec_heads: int = 8
    dropout: float = 0.0
    # pretext
    mask_ratio: float = 0.6
    diffusion_steps: int = 1000
    motion_intervals: int = 4  # kappa: stratified timesteps of the motion loss
    gamma_cen: float = 0.1
    lambda_mot: float = 1.0
    # Reconstruction target of the masked tokens: "position" (L_geo, Chamfer
    # distance on coordinates) or "velocity" (MSE on inter-frame displacements).
    recon_target: str = "position"
    # Probability of replacing the motion head's condition by a learned null
    # vector during training (condition dropout; no guidance at inference).
    motion_cond_dropout: float = 0.0
    # Pseudo-generator randomization of the Stage 1 input (dimp/pseudogen.py);
    # pseudo_gen_dro_eta > 0 weights L_mot over the families by GroupDRO.
    pseudo_gen: bool = False
    pseudo_gen_dro_eta: float = 0.0
    # Auxiliary behavioral tasks (dimp/aux_tasks.py).
    aux_tasks: tuple[str, ...] = ()
    motion_energy_aux_weight: float = 0.01
    region_motion_aux_weight: float = 0.01
    region_motion_exclude: tuple[str, ...] = ("jaw_contour",)
    eye_shape_aux_weight: float = 0.01
    # identity model (Stage 2): mean of the token features -> Linear
    embedding_dim: int = 256

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "DiMPConfig":
        """Builds the config from a checkpoint, accepting the keys of the
        research code only where they hold the setting this code implements."""
        unsupported = {
            key: value for key, value in values.items()
            if key in _LEGACY and _LEGACY[key] is not None and _normalize(value) not in _LEGACY[key]
        }
        if unsupported:
            raise ValueError(f"checkpoint uses settings not supported by this release: {unsupported}")
        names = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in values.items() if k in names})
        config.aux_tasks = tuple(config.aux_tasks)
        config.region_motion_exclude = tuple(config.region_motion_exclude)
        return config


def _normalize(value):
    return tuple(value) if isinstance(value, list) else value


# Keys of the research code and the values this release reproduces (None: ignored).
_LEGACY = {
    "mask_tube_length": {1},
    "point_encoder": {"flat"},
    "geo_loss": {"chamfer"},
    "objective": {"dimp"},
    "dec_mask_input": {"content"},
    "dec_cond": {"add"},
    "attn_pattern": {"full"},
    "region_adjacency": {()},
    "enc_mask_content": {"noisy"},
    "source_swap_weight": {0.0, 0},
    "motion_t_window": {()},
    "motion_lags": {(1,)},
    "source_adv_weight": {0.0, 0},
    "pooling": {"mean"},
    "motion_t_window_prob": None, "num_sources": None, "mode_aux_weight": None, "pose_aux_weight": None,
    "gaze_aux_weight": None, "motion_dynamics_aux_weight": None, "mode_invariant_aux_weight": None,
    "head_depth": None, "head_heads": None, "num_query_tokens": None, "head_mlp_ratio": None,
    "head_mlp_depth": None,
}


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.dropout = dropout
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = x if context is None else context
        batch, length, dim = x.shape
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            split(q), split(k), split(v), dropout_p=self.dropout if self.training else 0.0
        )
        return self.to_out(out.transpose(1, 2).reshape(batch, length, dim))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisMaskBlock(nn.Module):
    """Visible tokens self-attend; masked tokens cross-attend to the visible
    stream of the same layer (keys and values from visible tokens only)."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.vis_norm1 = nn.LayerNorm(dim)
        self.vis_attn = Attention(dim, heads, dropout)
        self.vis_norm2 = nn.LayerNorm(dim)
        self.vis_ff = FeedForward(dim, dropout)
        self.mask_norm_q = nn.LayerNorm(dim)
        self.mask_norm_kv = nn.LayerNorm(dim)
        self.mask_cross = Attention(dim, heads, dropout)
        self.mask_norm2 = nn.LayerNorm(dim)
        self.mask_ff = FeedForward(dim, dropout)

    def mask_stream(self) -> list[nn.Module]:
        return [self.mask_norm_q, self.mask_norm_kv, self.mask_cross, self.mask_norm2, self.mask_ff]

    def forward(
        self, visible: torch.Tensor, masked: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if masked is not None:
            masked = masked + self.mask_cross(self.mask_norm_q(masked), self.mask_norm_kv(visible))
            masked = masked + self.mask_ff(self.mask_norm2(masked))
        visible = visible + self.vis_attn(self.vis_norm1(visible))
        visible = visible + self.vis_ff(self.vis_norm2(visible))
        return visible, masked


class SelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class MotionNoiseHead(nn.Module):
    """Noise predictor of the motion diffusion, conditioned on the decoder
    output through a reset gate (as in the public DiMP release)."""

    def __init__(self, motion_dim: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.noise_proj = nn.Linear(motion_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.time_proj = nn.Linear(hidden, hidden)
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid()
        )
        self.fuse = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, motion_dim)

    def forward(
        self, noisy_motion: torch.Tensor, time_embedding: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        h_noise = self.noise_proj(noisy_motion)
        h_cond = self.cond_proj(condition)
        h_time = self.time_proj(time_embedding)[:, None]
        gate = self.reset_gate(torch.cat([h_noise, h_cond, h_time.expand_as(h_noise)], dim=-1))
        return self.out(self.fuse(gate * h_cond + (1 - gate) * h_noise + h_time))


def _init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MeshCore(nn.Module):
    """Token layout + dual-stream encoder; the only part kept after Stage 1."""

    def __init__(self, config: DiMPConfig) -> None:
        super().__init__()
        self.config = config
        self.layout = TubeLayout(config.num_groups, config.group_size, config.tube_frames, config.frames)
        flat = config.group_size * config.tube_frames * 2
        # phi^v: clean coordinates of visible tokens
        self.vis_embed = nn.Sequential(nn.Linear(flat, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        # phi^m: corrupted coordinates of masked tokens
        self.mask_embed = nn.Linear(flat, config.dim)
        # phi^enc: positional embedding of (centroid, time) + group embedding
        self.pos_mlp = nn.Sequential(nn.Linear(3, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.group_embedding = nn.Parameter(torch.zeros(config.num_groups, config.dim))
        self.blocks = nn.ModuleList(
            [VisMaskBlock(config.dim, config.heads, config.dropout) for _ in range(config.depth)]
        )
        self.norm_vis = nn.LayerNorm(config.dim)
        self.norm_mask = nn.LayerNorm(config.dim)
        _init(self)
        nn.init.trunc_normal_(self.group_embedding, std=0.02)

    def positions(self, centers: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """``centers (B, N, 2)``, ``token_ids (B, N)`` -> ``(B, N, dim)``."""
        chunk = self.layout.chunk_index[token_ids].float() / self.layout.chunks
        group = self.layout.group_index[token_ids]
        return self.pos_mlp(torch.cat([centers, chunk[..., None]], dim=-1)) + self.group_embedding[group]

    def encode(
        self, visible: torch.Tensor, masked: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for block in self.blocks:
            visible, masked = block(visible, masked)
        return self.norm_vis(visible), (None if masked is None else self.norm_mask(masked))

    def encode_all(self, mesh: torch.Tensor) -> torch.Tensor:
        """All tokens through the visible stream (Stage 2 and inference): ``(B, K, dim)``."""
        tokens = self.layout.tokens(mesh)
        ids = torch.arange(self.layout.num_tokens, device=mesh.device)[None].expand(mesh.shape[0], -1)
        embedded = self.vis_embed(tokens["points"].flatten(2)) + self.positions(tokens["centers"], ids)
        visible, _ = self.encode(embedded, None)
        return visible

    def mask_stream_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.mask_embed, self.norm_mask]
        for block in self.blocks:
            modules += block.mask_stream()
        return modules


def _gather(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    index = ids.view(*ids.shape, *([1] * (x.dim() - 2))).expand(*ids.shape, *x.shape[2:])
    return x.gather(1, index)


def chamfer_l2(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-frame two-sided squared-L2 Chamfer distance; ``(..., n, 2)`` points,
    ``valid (..., n)`` excludes padding from both sides. Returns ``(...)``."""
    distance = ((pred[..., :, None, :] - target[..., None, :, :]) ** 2).sum(-1)
    big = torch.finfo(distance.dtype).max / 4
    pair = valid[..., :, None] & valid[..., None, :]
    distance = distance.masked_fill(~pair, big)
    count = valid.sum(-1).clamp_min(1)
    pred_to_target = (distance.min(-1).values * valid).sum(-1) / count
    target_to_pred = (distance.min(-2).values * valid).sum(-1) / count
    return pred_to_target + target_to_pred


class DiMPPretrainer(nn.Module):
    """Stage 1: L_geo + gamma_cen L_cen + lambda_mot L_mot (+ auxiliary tasks)."""

    def __init__(self, config: DiMPConfig) -> None:
        super().__init__()
        if config.recon_target not in ("position", "velocity"):
            raise ValueError(f"unknown recon_target {config.recon_target!r}")
        self.config = config
        self.core = MeshCore(config)
        self.schedule = CosineSchedule(config.diffusion_steps)

        self.center_predictor = nn.Sequential(
            nn.Linear(config.dim, config.dim), nn.GELU(), nn.Linear(config.dim, 2)
        )
        self.dec_embed_vis = nn.Linear(config.dim, config.dec_dim)
        self.dec_embed_mask = nn.Linear(config.dim, config.dec_dim)
        # phi^dec: decoder positional embedding + group embedding
        self.dec_pos_mlp = nn.Sequential(
            nn.Linear(3, config.dec_dim), nn.GELU(), nn.Linear(config.dec_dim, config.dec_dim)
        )
        self.dec_group_embedding = nn.Parameter(torch.zeros(config.num_groups, config.dec_dim))
        self.dec_time = TimeEmbedding(config.dec_dim)
        self.dec_blocks = nn.ModuleList(
            [SelfBlock(config.dec_dim, config.dec_heads, config.dropout) for _ in range(config.dec_depth)]
        )
        self.dec_norm = nn.LayerNorm(config.dec_dim)
        steps = config.tube_frames - 1 if config.recon_target == "velocity" else config.tube_frames
        self.recon_head = nn.Linear(config.dec_dim, config.group_size * steps * 2)
        motion_dim = config.group_size * (config.tube_frames - 1) * 2
        self.motion_time = TimeEmbedding(config.dec_dim)
        self.motion_head = MotionNoiseHead(motion_dim, config.dec_dim, config.dec_dim)
        if config.motion_cond_dropout > 0:
            self.motion_null_cond = nn.Parameter(torch.zeros(1, 1, config.dec_dim))
        if config.pseudo_gen:
            from .pseudogen import NUM_FAMILIES, landmark_parts

            part, parts = landmark_parts()
            self.register_buffer("pg_part", part, persistent=False)
            self.pg_parts = parts
            self.register_buffer("dro_q", torch.full((NUM_FAMILIES,), 1.0 / NUM_FAMILIES), persistent=False)
        for module in (
            self.center_predictor, self.dec_embed_vis, self.dec_embed_mask, self.dec_pos_mlp,
            self.dec_blocks, self.recon_head, self.motion_head,
        ):
            _init(module)
        nn.init.trunc_normal_(self.dec_group_embedding, std=0.02)
        self.aux_tasks = build_aux_tasks(config)

    @torch.no_grad()
    def pooled_features(self, mesh: torch.Tensor) -> torch.Tensor:
        """Head-free probe used for monitoring: mean of the encoder outputs, L2-normalized."""
        return F.normalize(self.core.encode_all(mesh).mean(dim=1).float(), dim=-1)

    def decoder_positions(self, centers: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        layout = self.core.layout
        chunk = layout.chunk_index[token_ids].float() / layout.chunks
        group = layout.group_index[token_ids]
        return self.dec_pos_mlp(torch.cat([centers, chunk[..., None]], dim=-1)) + self.dec_group_embedding[group]

    def _add_aux(
        self, output: dict, mesh: torch.Tensor, tokens: dict, z_vis: torch.Tensor, ids_vis: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Adds the auxiliary task losses, computed from the visible-token features."""
        if not self.aux_tasks:
            return output
        config = self.config
        motion_vis = _gather(tokens["motion"].flatten(2), ids_vis)
        extra = {
            "z_vis": z_vis,
            "motion_vis": motion_vis.view(*motion_vis.shape[:2], config.group_size, config.tube_frames - 1, 2),
            "point_valid_vis": self.core.layout.point_valid()[ids_vis],
        }
        ctx = AuxTaskContext(mesh=mesh, pooled=z_vis.mean(dim=1), config=config, extra=extra)
        for name, task in self.aux_tasks.items():
            result = task(ctx)
            output["loss"] = output["loss"] + aux_task_weight(config, name) * result["loss"]
            output[f"aux_{name}_loss"] = result["loss"].detach()
        return output

    def forward(
        self, mesh: torch.Tensor, *, t_c: torch.Tensor | None = None, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        config, layout = self.config, self.core.layout
        batch, device = mesh.shape[0], mesh.device
        num_tokens = layout.num_tokens
        num_masked = int(round(config.mask_ratio * num_tokens))

        def rand(*shape: int) -> torch.Tensor:
            return torch.rand(*shape, device=device, generator=generator)

        def randn(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, device=device, generator=generator)

        family = None
        if config.pseudo_gen and self.training:
            from .pseudogen import NUM_FAMILIES, apply_pseudo_generator

            family = (rand(batch) * NUM_FAMILIES).long().clamp(max=NUM_FAMILIES - 1)
            with torch.no_grad():
                mesh = apply_pseudo_generator(mesh, family, self.pg_part, self.pg_parts, rand, randn)
        tokens = layout.tokens(mesh)

        order = rand(batch, num_tokens).argsort(dim=1)
        ids_mask, ids_vis = order[:, :num_masked], order[:, num_masked:]
        points_flat = tokens["points"].flatten(2)  # (B, K, n*l*2)
        point_valid = layout.point_valid()  # (K, n)
        points_v = _gather(points_flat, ids_vis)
        points_m0 = _gather(points_flat, ids_mask)
        centers_v = _gather(tokens["centers"], ids_vis)
        centers_m0 = _gather(tokens["centers"], ids_mask)

        # one shared timestep t_c per sample; centroids and coordinates get independent noise
        if t_c is None:
            t_c = (rand(batch) * config.diffusion_steps).long().clamp(max=config.diffusion_steps - 1) + 1
        valid_m = point_valid[ids_mask]  # (B, Km, n)
        valid_m_flat = valid_m[..., None, None].expand(-1, -1, -1, config.tube_frames, 2).flatten(2)
        centers_mt = self.schedule.add_noise(centers_m0, t_c, randn(*centers_m0.shape))
        points_mt = self.schedule.add_noise(points_m0, t_c, randn(*points_m0.shape)) * valid_m_flat

        # encoder: clean visible tokens, corrupted masked tokens at their corrupted centroids
        enc_vis = self.core.vis_embed(points_v) + self.core.positions(centers_v, ids_vis)
        content_m = self.core.mask_embed(points_mt)
        enc_mask = content_m + self.core.positions(centers_mt, ids_mask)
        z_vis, z_mask = self.core.encode(enc_vis, enc_mask)
        centers_hat = self.center_predictor(z_mask)
        loss_cen = F.mse_loss(centers_hat.float(), centers_m0.float())

        # decoder: masked tokens at the stop-gradient predicted centroids, plus tau(t_c)
        content = torch.cat([self.dec_embed_vis(z_vis), self.dec_embed_mask(content_m)], dim=1)
        position = torch.cat(
            [self.decoder_positions(centers_v, ids_vis), self.decoder_positions(centers_hat.detach(), ids_mask)], dim=1
        )
        sequence = content + position + self.dec_time(t_c)[:, None]
        for block in self.dec_blocks:
            sequence = block(sequence)
        z_dec_mask = self.dec_norm(sequence)[:, ids_vis.shape[1]:]

        motion0 = _gather(tokens["motion"].flatten(2), ids_mask).float()
        motion_valid = valid_m[..., None, None].expand(-1, -1, -1, config.tube_frames - 1, 2).flatten(2).float()
        if config.recon_target == "velocity":
            pred = self.recon_head(z_dec_mask).float()
            scale = ((motion0 ** 2 * motion_valid).sum() / motion_valid.sum()).sqrt().clamp_min(1e-6).detach()
            loss_geo = (((pred - motion0 / scale) ** 2) * motion_valid).sum() / motion_valid.sum()
        else:
            pred = self.recon_head(z_dec_mask).float().view(batch, num_masked, config.group_size, config.tube_frames, 2)
            target = points_m0.float().view_as(pred)
            frame_valid = valid_m[:, :, None, :].expand(-1, -1, config.tube_frames, -1)
            loss_geo = chamfer_l2(pred.permute(0, 1, 3, 2, 4), target.permute(0, 1, 3, 2, 4), frame_valid).mean()

        # L_mot: noise prediction on the displacements at kappa stratified timesteps
        cond = z_dec_mask
        if self.training and config.motion_cond_dropout > 0:
            drop = rand(batch, cond.shape[1]) < config.motion_cond_dropout
            cond = torch.where(drop[..., None], self.motion_null_cond.to(cond.dtype).expand_as(cond), cond)
        loss_mot = motion0.new_zeros(())
        for low, high in self.schedule.stratified_intervals(config.motion_intervals):
            t_i = (rand(batch) * (high - low + 1)).long().clamp(max=high - low) + low
            eps = randn(*motion0.shape)
            noisy = self.schedule.add_noise(motion0, t_i, eps) * motion_valid
            eps_hat = self.motion_head(noisy, self.motion_time(t_i), cond).float()
            per_sample = (((eps_hat - eps) ** 2) * motion_valid).flatten(1).sum(1) / motion_valid.flatten(1).sum(1).clamp_min(1)
            loss_mot = loss_mot + per_sample
        loss_mot = loss_mot / config.motion_intervals  # (B,)
        if family is not None and config.pseudo_gen_dro_eta > 0:
            # GroupDRO over the pseudo-generator families
            groups = torch.stack([
                loss_mot[family == g].mean() if (family == g).any() else loss_mot.new_zeros(())
                for g in range(self.dro_q.numel())
            ])
            present = torch.stack([(family == g).any() for g in range(self.dro_q.numel())]).float()
            with torch.no_grad():
                q = self.dro_q * torch.exp(config.pseudo_gen_dro_eta * groups.detach() * present)
                self.dro_q.copy_(q / q.sum())
            weights = self.dro_q * present
            loss_mot = (weights * groups).sum() / weights.sum()
        else:
            loss_mot = loss_mot.mean()

        loss = loss_geo + config.gamma_cen * loss_cen + config.lambda_mot * loss_mot
        output = {"loss": loss, "geo": loss_geo.detach(), "cen": loss_cen.detach(), "mot": loss_mot.detach()}
        return self._add_aux(output, mesh, tokens, z_vis, ids_vis)


class DiMPIdentity(nn.Module):
    """Stage 2 / inference: all tokens through the visible stream, mean, Linear, L2 norm."""

    def __init__(self, config: DiMPConfig) -> None:
        super().__init__()
        self.config = config
        self.core = MeshCore(config)
        for module in self.core.mask_stream_modules():  # unused after Stage 1
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.embedding_projection = nn.Linear(config.dim, config.embedding_dim)
        _init(self.embedding_projection)

    def forward(self, mesh: torch.Tensor) -> torch.Tensor:
        feature = self.core.encode_all(mesh).mean(dim=1)
        return F.normalize(self.embedding_projection(feature).float(), dim=-1)
