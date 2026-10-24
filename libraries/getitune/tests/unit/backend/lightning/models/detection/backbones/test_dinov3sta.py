# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from getitune.backend.lightning.models.common.backbones.dinov3 import (
    DinoVisionTransformer,
    _map_legacy_dinov3_state_dict,
)
from getitune.backend.lightning.models.detection.backbones.dinov3sta import (
    DINOv3STAsModule,
)


@pytest.mark.parametrize(
    ("name", "gated_mlp"),
    [("dinov3_vits16", False), ("dinov3_vits16plus", True)],
)
def test_create_dinov3_backbone(name: str, gated_mlp: bool) -> None:
    backbone = DinoVisionTransformer(name=name, interaction_indexes=[5, 8, 11])

    assert backbone.config.hidden_size == 384
    assert backbone.config.intermediate_size == 1536
    assert backbone.config.num_hidden_layers == 12
    assert backbone.config.out_indices == [6, 9, 12]
    assert backbone.config.use_gated_mlp is gated_mlp
    for projection in ("q_proj", "k_proj", "v_proj"):
        assert not any(f"attention.{projection}.bias" in key for key in backbone.state_dict())


def test_map_legacy_dinov3_state_dict() -> None:
    prefix = "model.backbone.dinov3."
    qkv_weight = torch.arange(36).reshape(12, 3)
    qkv_bias = torch.arange(12)
    fc1_weight = torch.arange(32).reshape(8, 4)
    fc2_weight = torch.arange(32).reshape(4, 8)
    state_dict = {
        f"{prefix}blocks.2.attn.qkv.weight": qkv_weight,
        f"{prefix}blocks.2.attn.qkv.bias": qkv_bias,
        f"{prefix}blocks.2.attn.qkv.bias_mask": torch.zeros(12),
        f"{prefix}blocks.2.attn.proj.weight": torch.ones(4, 4),
        f"{prefix}blocks.2.ls1.gamma": torch.ones(4),
        f"{prefix}blocks.2.mlp.fc1.weight": fc1_weight,
        f"{prefix}blocks.2.mlp.fc2.weight": fc2_weight,
        f"{prefix}rope_embed.periods": torch.ones(2),
        f"{prefix}mask_token": torch.zeros(1, 4),
    }

    _map_legacy_dinov3_state_dict(state_dict, "model.backbone.")

    assert torch.equal(state_dict[f"{prefix}model.layer.2.attention.q_proj.weight"], qkv_weight[:4])
    assert torch.equal(state_dict[f"{prefix}model.layer.2.attention.k_proj.weight"], qkv_weight[4:8])
    assert torch.equal(state_dict[f"{prefix}model.layer.2.attention.v_proj.weight"], qkv_weight[8:])
    assert f"{prefix}model.layer.2.attention.q_proj.bias" not in state_dict
    assert f"{prefix}model.layer.2.attention.o_proj.weight" in state_dict
    assert f"{prefix}model.layer.2.layer_scale1.lambda1" in state_dict
    assert torch.equal(state_dict[f"{prefix}model.layer.2.mlp.up_proj.weight"], fc1_weight)
    assert torch.equal(state_dict[f"{prefix}model.layer.2.mlp.down_proj.weight"], fc2_weight)
    assert state_dict[f"{prefix}embeddings.mask_token"].shape == (1, 1, 4)
    assert f"{prefix}rope_embed.periods" not in state_dict
    assert f"{prefix}blocks.2.attn.qkv.bias_mask" not in state_dict


def test_map_legacy_dinov3_swiglu_state_dict() -> None:
    prefix = "backbone.dinov3."
    legacy_weights = {name: torch.ones(4, 4) for name in ("w1", "w2", "w3")}
    state_dict = {f"{prefix}blocks.0.mlp.{name}.weight": value for name, value in legacy_weights.items()}

    _map_legacy_dinov3_state_dict(state_dict, "backbone.")

    assert torch.equal(state_dict[f"{prefix}model.layer.0.mlp.gate_proj.weight"], legacy_weights["w1"])
    assert torch.equal(state_dict[f"{prefix}model.layer.0.mlp.up_proj.weight"], legacy_weights["w2"])
    assert torch.equal(state_dict[f"{prefix}model.layer.0.mlp.down_proj.weight"], legacy_weights["w3"])


def test_map_legacy_dinov3_rejects_active_qkv_bias() -> None:
    state_dict = {"backbone.dinov3.blocks.0.attn.qkv.bias": torch.ones(12)}

    with pytest.raises(ValueError, match="active DINOv3 QKV bias"):
        _map_legacy_dinov3_state_dict(state_dict, "backbone.")


def test_dinov3_sta_forward_shapes() -> None:
    backbone = DINOv3STAsModule(
        name="dinov3_vits16",
        interaction_indexes=[5, 8, 11],
        conv_inplane=2,
        hidden_dim=16,
    ).eval()

    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 64, 64))

    assert [tuple(feature.shape) for feature in features] == [
        (1, 16, 8, 8),
        (1, 16, 4, 4),
        (1, 16, 2, 2),
    ]
    assert all(torch.isfinite(feature).all() for feature in features)
