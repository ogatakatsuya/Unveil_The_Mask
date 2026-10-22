from __future__ import annotations

import pytest
import torch

from dimp.data import prepare_mesh
from dimp.model import DiMPConfig, DiMPIdentity, DiMPPretrainer
from dimp.objective import supervised_contrastive_loss
from dimp.tubes import build_groups

FRAMES, TUBE, GROUPS = 16, 4, 4


def make_config(**overrides) -> tuple[DiMPConfig, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    groups, valid = build_groups(torch.randn(106, 2), GROUPS)
    values = dict(
        frames=FRAMES, tube_frames=TUBE, num_groups=GROUPS, group_size=groups.shape[1],
        dim=32, depth=2, heads=4, dec_dim=32, dec_depth=1, dec_heads=4, embedding_dim=12,
        aux_tasks=("motion_energy", "region_motion", "eye_shape"),
    )
    values.update(overrides)
    return DiMPConfig(**values), groups, valid


def mesh_batch(batch: int = 4) -> torch.Tensor:
    return torch.stack([prepare_mesh(torch.randn(FRAMES, 106, 2), FRAMES, training=True) for _ in range(batch)])


def test_prepare_mesh_removes_static_shape() -> None:
    clip = torch.randn(1, 106, 2).expand(FRAMES, -1, -1) * 3 + torch.randn(FRAMES, 1, 2) * 0.1
    mesh = prepare_mesh(clip + torch.randn(FRAMES, 106, 2) * 0.01, FRAMES, training=False)
    assert mesh.shape == (FRAMES, 106, 2)
    torch.testing.assert_close(mesh.mean(dim=0), torch.zeros(106, 2), atol=1e-3, rtol=0)
    assert abs(float(mesh.norm(dim=-1).mean()) - 1.0) < 1e-4


@pytest.mark.parametrize("overrides", [
    {},
    {"recon_target": "velocity"},
    {"lambda_mot": 10.0, "motion_cond_dropout": 0.15},
    {"lambda_mot": 10.0, "pseudo_gen": True, "pseudo_gen_dro_eta": 0.01},
])
def test_pretraining_losses_are_finite(overrides: dict) -> None:
    config, groups, valid = make_config(**overrides)
    model = DiMPPretrainer(config)
    model.core.layout.set_groups(groups, valid)
    losses = model(mesh_batch())
    assert torch.isfinite(losses["loss"])
    assert {"geo", "cen", "mot", "aux_motion_energy_loss", "aux_region_motion_loss", "aux_eye_shape_loss"} <= set(losses)
    losses["loss"].backward()
    assert model.core.vis_embed[0].weight.grad is not None


def test_identity_model_uses_the_pretrained_encoder() -> None:
    config, groups, valid = make_config()
    pretrainer = DiMPPretrainer(config)
    pretrainer.core.layout.set_groups(groups, valid)
    identity = DiMPIdentity(config)
    identity.core.load_state_dict(pretrainer.core.state_dict())
    mesh = mesh_batch(8)
    embeddings = identity(mesh)
    assert embeddings.shape == (8, config.embedding_dim)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(8))
    loss = supervised_contrastive_loss(embeddings, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    assert torch.isfinite(loss)


def test_config_from_research_checkpoint() -> None:
    values = DiMPConfig().to_dict() | {"attn_pattern": "full", "motion_lags": [1], "pooling": "mean", "gaze_aux_weight": 0.1}
    assert DiMPConfig.from_dict(values) == DiMPConfig()
    with pytest.raises(ValueError):
        DiMPConfig.from_dict(values | {"attn_pattern": "causal"})
