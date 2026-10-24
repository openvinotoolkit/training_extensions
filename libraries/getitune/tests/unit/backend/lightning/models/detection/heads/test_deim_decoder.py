# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DEIM Transformer Decoder."""

from __future__ import annotations

import copy

import pytest
import torch

from getitune.backend.lightning.models.detection.heads.deim_decoder import (
    DEIMTransformer,
    DEIMTransformerModule,
    TransformerDecoder,
    TransformerDecoderLayer,
)


class TestTransformerDecoderLayer:
    """Test class for TransformerDecoderLayer."""

    @pytest.fixture
    def decoder_layer(self):
        """Create a basic decoder layer."""
        return TransformerDecoderLayer(
            d_model=256,
            n_head=8,
            dim_feedforward=1024,
            dropout=0.0,
            n_levels=3,
            n_points=4,
        )

    @pytest.fixture
    def decoder_layer_with_gateway(self):
        """Create a decoder layer with gateway enabled."""
        return TransformerDecoderLayer(
            d_model=256,
            n_head=8,
            dim_feedforward=1024,
            dropout=0.0,
            n_levels=3,
            n_points=4,
            use_gateway=True,
        )

    @pytest.fixture
    def decoder_layer_with_scale(self):
        """Create a decoder layer with layer scale."""
        return TransformerDecoderLayer(
            d_model=256,
            n_head=8,
            dim_feedforward=1024,
            dropout=0.0,
            n_levels=3,
            n_points=[3, 6, 3],
            layer_scale=2.0,
        )

    def test_decoder_layer_init(self, decoder_layer):
        """Test decoder layer initialization."""
        assert isinstance(decoder_layer, TransformerDecoderLayer)
        assert decoder_layer.use_gateway is False
        # Check for memory-efficient self-attention components
        assert hasattr(decoder_layer, "qkv_proj")
        assert hasattr(decoder_layer, "out_proj")
        assert hasattr(decoder_layer, "cross_attn")
        assert hasattr(decoder_layer, "swish_ffn")

    def test_decoder_layer_with_gateway_init(self, decoder_layer_with_gateway):
        """Test decoder layer with gateway initialization."""
        assert decoder_layer_with_gateway.use_gateway is True
        assert hasattr(decoder_layer_with_gateway, "gateway")

    def test_decoder_layer_forward(self, decoder_layer):
        """Test decoder layer forward pass."""
        batch_size = 2
        num_queries = 300
        hidden_dim = 256
        spatial_shapes = [[40, 40], [20, 20], [10, 10]]

        target = torch.randn(batch_size, num_queries, hidden_dim)
        reference_points = torch.rand(batch_size, num_queries, 1, 4)

        # Create value tuple for each level
        values = tuple(torch.randn(batch_size, 8, hidden_dim // 8, h * w) for h, w in spatial_shapes)

        output = decoder_layer(
            target=target,
            reference_points=reference_points,
            value=values,
            spatial_shapes=spatial_shapes,
        )

        assert output.shape == target.shape
        assert not torch.isnan(output).any()

    def test_with_pos_embed(self, decoder_layer):
        """Test position embedding addition."""
        tensor = torch.randn(2, 100, 256)
        pos = torch.randn(2, 100, 256)

        # With position embedding
        result = decoder_layer.with_pos_embed(tensor, pos)
        assert torch.allclose(result, tensor + pos)

        # Without position embedding
        result_no_pos = decoder_layer.with_pos_embed(tensor, None)
        assert torch.allclose(result_no_pos, tensor)

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_self_attention_matches_multihead_attention(self, decoder_layer, use_mask):
        """Self-attention must match nn.MultiheadAttention(q, k, value=target) semantics.

        Q/K come from the position-embedded queries, V from the raw target.
        """
        torch.manual_seed(0)
        batch_size, num_queries, hidden_dim, n_head = 2, 50, 256, 8
        target = torch.randn(batch_size, num_queries, hidden_dim)
        pos = torch.randn(batch_size, num_queries, hidden_dim)
        qk = decoder_layer.with_pos_embed(target, pos)

        attn_mask = None
        if use_mask:
            attn_mask = torch.zeros(num_queries, num_queries, dtype=torch.bool)
            attn_mask[: num_queries // 2, num_queries // 2 :] = True

        mha = torch.nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)
        mha.in_proj_weight.data.copy_(decoder_layer.qkv_proj.weight)
        mha.in_proj_bias.data.copy_(decoder_layer.qkv_proj.bias)
        mha.out_proj.load_state_dict(decoder_layer.out_proj.state_dict())
        mha.eval()

        with torch.no_grad():
            expected, _ = mha(qk, qk, target, attn_mask=attn_mask, need_weights=False)
            actual = decoder_layer._self_attention(qk, target, attn_mask=attn_mask)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    @staticmethod
    def _forward_inputs(batch_size: int = 2, num_queries: int = 30, hidden_dim: int = 256, n_head: int = 8) -> dict:
        """Build decoder-layer forward inputs with a nonzero positional embedding."""
        spatial_shapes = [[8, 8], [4, 4], [2, 2]]
        return {
            "target": torch.randn(batch_size, num_queries, hidden_dim),
            "reference_points": torch.rand(batch_size, num_queries, 1, 4),
            "value": tuple(torch.randn(batch_size, n_head, hidden_dim // n_head, h * w) for h, w in spatial_shapes),
            "spatial_shapes": spatial_shapes,
            "query_pos_embed": torch.randn(batch_size, num_queries, hidden_dim) * 2.0,
        }

    def test_forward_projects_value_from_raw_target(self, decoder_layer, monkeypatch):
        """forward must feed Q/K with target + pos and V with the raw target (no positional embedding)."""
        torch.manual_seed(0)
        inputs = self._forward_inputs()
        target, pos = inputs["target"], inputs["query_pos_embed"]
        captured: dict[str, torch.Tensor] = {}
        original = decoder_layer._self_attention

        def spy(qk: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
            captured["qk"], captured["v"] = qk.clone(), v.clone()
            return original(qk, v, attn_mask=attn_mask)

        monkeypatch.setattr(decoder_layer, "_self_attention", spy)
        with torch.no_grad():
            decoder_layer(**inputs)

        torch.testing.assert_close(captured["qk"], target + pos)
        torch.testing.assert_close(captured["v"], target)
        assert not torch.allclose(captured["v"], captured["qk"])

    def test_forward_matches_multihead_attention_reference(self, decoder_layer):
        """End-to-end layer output must match a reference whose self-attention is nn.MultiheadAttention.

        The reference uses ``mha(target + pos, target + pos, value=target)`` like upstream DEIMv2, so
        projecting V from the position-embedded queries would make the outputs diverge.
        """
        torch.manual_seed(0)
        decoder_layer.eval()
        inputs = self._forward_inputs()
        target, pos = inputs["target"], inputs["query_pos_embed"]

        mha = torch.nn.MultiheadAttention(256, 8, batch_first=True)
        mha.in_proj_weight.data.copy_(decoder_layer.qkv_proj.weight)
        mha.in_proj_bias.data.copy_(decoder_layer.qkv_proj.bias)
        mha.out_proj.load_state_dict(decoder_layer.out_proj.state_dict())
        mha.eval()

        reference_layer = copy.deepcopy(decoder_layer)

        def reference_self_attention(
            qk: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Independent of what forward passes in: always use upstream semantics.
            out, _ = mha(target + pos, target + pos, target, need_weights=False)
            return out

        reference_layer._self_attention = reference_self_attention

        with torch.no_grad():
            actual = decoder_layer(**inputs)
            expected = reference_layer(**inputs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class TestDEIMTransformerModule:
    """Test class for DEIMTransformerModule."""

    @pytest.fixture
    def deim_transformer(self):
        """Create a basic DEIM transformer module."""
        return DEIMTransformerModule(
            num_classes=10,
            hidden_dim=256,
            num_queries=100,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=[3, 6, 3],
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.0,
            num_denoising=50,
            eval_spatial_size=(640, 640),
            reg_max=32,
        )

    @pytest.fixture
    def deim_transformer_minimal(self):
        """Create a minimal DEIM transformer for faster testing."""
        return DEIMTransformerModule(
            num_classes=5,
            hidden_dim=128,
            num_queries=50,
            feat_channels=[128, 128, 128],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=[2, 4, 2],
            nhead=4,
            num_layers=1,
            dim_feedforward=256,
            dropout=0.0,
            num_denoising=0,
            eval_spatial_size=(320, 320),
            reg_max=16,
        )

    @pytest.fixture
    def targets(self):
        """Create sample targets for training."""
        return [
            {
                "boxes": torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
                "labels": torch.tensor([1, 0]),
            },
            {
                "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
                "labels": torch.tensor([2]),
            },
        ]

    def test_deim_transformer_init(self, deim_transformer):
        """Test DEIM transformer initialization."""
        assert isinstance(deim_transformer, DEIMTransformerModule)
        assert deim_transformer.num_classes == 10
        assert deim_transformer.hidden_dim == 256
        assert deim_transformer.num_queries == 100
        assert deim_transformer.num_levels == 3
        assert deim_transformer.reg_max == 32
        assert deim_transformer.aux_loss is True

    def test_deim_transformer_components(self, deim_transformer):
        """Test that all components are properly initialized."""
        # Check input projection
        assert hasattr(deim_transformer, "input_proj")
        assert len(deim_transformer.input_proj) == 3

        # Check decoder
        assert hasattr(deim_transformer, "decoder")
        assert isinstance(deim_transformer.decoder, TransformerDecoder)

        # Check heads
        assert hasattr(deim_transformer, "enc_score_head")
        assert hasattr(deim_transformer, "enc_bbox_head")
        assert hasattr(deim_transformer, "dec_score_head")
        assert hasattr(deim_transformer, "dec_bbox_head")
        assert hasattr(deim_transformer, "pre_bbox_head")
        assert hasattr(deim_transformer, "query_pos_head")

        # Check denoising embedding
        assert hasattr(deim_transformer, "denoising_class_embed")

        # Check integral for FDR
        assert hasattr(deim_transformer, "integral")

    def test_deim_transformer_forward_training(self, deim_transformer_minimal, targets):
        """Test DEIM transformer forward pass in training mode."""
        deim_transformer_minimal.train()

        feats = [
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]

        output = deim_transformer_minimal(feats, targets)

        assert isinstance(output, dict)
        assert "pred_logits" in output
        assert "pred_boxes" in output
        assert "pred_corners" in output
        assert "ref_points" in output

        # Check output shapes
        num_queries = deim_transformer_minimal.num_queries
        num_classes = deim_transformer_minimal.num_classes
        assert output["pred_logits"].shape == (2, num_queries, num_classes)
        assert output["pred_boxes"].shape == (2, num_queries, 4)

    def test_deim_transformer_forward_eval(self, deim_transformer_minimal):
        """Test DEIM transformer forward pass in eval mode."""
        deim_transformer_minimal.eval()

        feats = [
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 128, 20, 20),
            torch.randn(1, 128, 10, 10),
        ]

        output = deim_transformer_minimal(feats)

        assert isinstance(output, dict)
        assert "pred_logits" in output
        assert "pred_boxes" in output

        # In eval mode, should not have training-specific outputs
        assert "pred_corners" not in output
        assert "aux_outputs" not in output

    def test_deim_transformer_forward_explain_mode(self, deim_transformer_minimal):
        """Test DEIM transformer forward pass with explain mode."""
        deim_transformer_minimal.eval()

        feats = [
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 128, 20, 20),
            torch.randn(1, 128, 10, 10),
        ]

        output = deim_transformer_minimal(feats, explain_mode=True)

        assert isinstance(output, dict)
        assert "raw_logits" in output

    def test_deim_transformer_aux_loss(self, deim_transformer_minimal, targets):
        """Test auxiliary loss outputs."""
        deim_transformer_minimal.train()

        feats = [
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]

        output = deim_transformer_minimal(feats, targets)

        # Check auxiliary outputs exist
        if deim_transformer_minimal.aux_loss:
            assert "aux_outputs" in output or deim_transformer_minimal.num_layers == 1
            assert "enc_aux_outputs" in output
            assert "pre_outputs" in output

    def test_generate_anchors(self, deim_transformer_minimal):
        """Test anchor generation."""
        spatial_shapes = [[40, 40], [20, 20], [10, 10]]

        anchors, valid_mask = deim_transformer_minimal._generate_anchors(
            spatial_shapes=spatial_shapes,
            device="cpu",
        )

        # Check anchor shape: should be [1, total_anchors, 4]
        total_anchors = sum(h * w for h, w in spatial_shapes)
        assert anchors.shape == (1, total_anchors, 4)
        assert valid_mask.shape == (1, total_anchors, 1)

    def test_get_encoder_input(self, deim_transformer_minimal):
        """Test encoder input preparation."""
        feats = [
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]

        feat_flatten, spatial_shapes = deim_transformer_minimal._get_encoder_input(feats)

        # Check flattened features
        total_tokens = 40 * 40 + 20 * 20 + 10 * 10
        assert feat_flatten.shape == (2, total_tokens, 128)

        # Check spatial shapes
        assert spatial_shapes == [[40, 40], [20, 20], [10, 10]]

    def test_select_topk(self, deim_transformer_minimal):
        """Test top-k query selection."""
        batch_size = 2
        num_tokens = 1000
        hidden_dim = 128
        num_classes = 5
        topk = 50

        memory = torch.randn(batch_size, num_tokens, hidden_dim)
        outputs_logits = torch.randn(batch_size, num_tokens, num_classes)
        outputs_anchors = torch.randn(batch_size, num_tokens, 4)

        topk_memory, topk_logits, topk_anchors = deim_transformer_minimal._select_topk(
            memory, outputs_logits, outputs_anchors, topk
        )

        assert topk_memory.shape == (batch_size, topk, hidden_dim)
        assert topk_anchors.shape == (batch_size, topk, 4)
        # topk_logits is None in eval mode
        if deim_transformer_minimal.training:
            assert topk_logits.shape == (batch_size, topk, num_classes)

    def test_convert_to_deploy(self, deim_transformer_minimal):
        """Test deployment conversion."""
        deim_transformer_minimal.convert_to_deploy()

        # After conversion, some heads should be Identity
        eval_idx = deim_transformer_minimal.eval_idx
        for i, head in enumerate(deim_transformer_minimal.dec_score_head):
            if i < eval_idx:
                assert isinstance(head, torch.nn.Identity)

    def test_input_proj_identity(self):
        """Test input projection with matching dimensions."""
        transformer = DEIMTransformerModule(
            num_classes=5,
            hidden_dim=256,
            feat_channels=[256, 256, 256],  # Same as hidden_dim
            num_layers=1,
            num_denoising=0,
        )

        # When feat_channels == hidden_dim, should use Identity
        for proj in transformer.input_proj:
            assert isinstance(proj, torch.nn.Identity)

    def test_input_proj_conv(self):
        """Test input projection with different dimensions."""
        transformer = DEIMTransformerModule(
            num_classes=5,
            hidden_dim=256,
            feat_channels=[128, 128, 128],  # Different from hidden_dim
            num_layers=1,
            num_denoising=0,
        )

        # When feat_channels != hidden_dim, should use Conv projection
        for proj in transformer.input_proj:
            assert isinstance(proj, torch.nn.Sequential)

    def test_validation_errors(self):
        """Test that validation errors are raised correctly."""
        # feat_channels > num_levels should raise error
        with pytest.raises(ValueError, match="feat_channels.*must be <= num_levels"):
            DEIMTransformerModule(
                num_classes=5,
                feat_channels=[256, 256, 256, 256],
                num_levels=3,
            )

        # feat_strides length mismatch should raise error
        with pytest.raises(ValueError, match="feat_strides.*must match feat_channels"):
            DEIMTransformerModule(
                num_classes=5,
                feat_channels=[256, 256, 256],
                feat_strides=[8, 16],  # Mismatch
            )


class TestDEIMTransformerFactory:
    """Test class for DEIMTransformer factory."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "deimv2_x",
            "deimv2_l",
            "deimv2_m",
            "deimv2_s",
        ],
    )
    def test_factory_creates_correct_model(self, model_name):
        """Test that factory creates correct model variants."""
        transformer = DEIMTransformer(
            model_name=model_name,
            num_classes=80,
            eval_spatial_size=(640, 640),
        )

        assert isinstance(transformer, DEIMTransformerModule)
        assert transformer.num_classes == 80
        assert transformer.eval_spatial_size == (640, 640)

    def test_factory_config_deimv2_x(self):
        """Test DEIMv2-X configuration."""
        transformer = DEIMTransformer(
            model_name="deimv2_x",
            num_classes=10,
        )

        assert transformer.hidden_dim == 256
        assert transformer.num_layers == 6

    def test_factory_config_deimv2_l(self):
        """Test DEIMv2-L configuration."""
        transformer = DEIMTransformer(
            model_name="deimv2_l",
            num_classes=10,
        )

        assert transformer.hidden_dim == 224
        assert transformer.num_layers == 4

    def test_factory_config_deimv2_m(self):
        """Test DEIMv2-M configuration."""
        transformer = DEIMTransformer(
            model_name="deimv2_m",
            num_classes=10,
        )

        assert transformer.hidden_dim == 256
        assert transformer.num_layers == 4

    def test_factory_config_deimv2_s(self):
        """Test DEIMv2-S configuration."""
        transformer = DEIMTransformer(
            model_name="deimv2_s",
            num_classes=10,
        )

        assert transformer.hidden_dim == 192
        assert transformer.num_layers == 4

    def test_factory_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(KeyError):
            DEIMTransformer(
                model_name="invalid_model",
                num_classes=10,
            )


class TestTransformerDecoder:
    """Test class for TransformerDecoder."""

    @pytest.fixture
    def decoder(self):
        """Create a basic transformer decoder."""
        decoder_layer = TransformerDecoderLayer(
            d_model=128,
            n_head=4,
            dim_feedforward=256,
            n_levels=3,
            n_points=4,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            d_model=128,
            n_head=4,
            dim_feedforward=256,
            n_levels=3,
            n_points=4,
            layer_scale=2.0,
        )
        up = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        reg_scale = torch.nn.Parameter(torch.tensor([4.0]), requires_grad=False)

        return TransformerDecoder(
            hidden_dim=128,
            decoder_layer=decoder_layer,
            decoder_layer_wide=decoder_layer_wide,
            num_layers=2,
            num_head=4,
            reg_max=16,
            reg_scale=reg_scale,
            up=up,
            eval_idx=-1,
            layer_scale=2,
        )

    def test_decoder_init(self, decoder):
        """Test decoder initialization."""
        assert isinstance(decoder, TransformerDecoder)
        assert decoder.hidden_dim == 128
        assert decoder.num_layers == 2
        assert len(decoder.layers) == 2
        assert len(decoder.lqe_layers) == 2

    def test_decoder_convert_to_deploy(self, decoder):
        """Test decoder deployment conversion."""
        original_num_layers = len(decoder.layers)
        decoder.convert_to_deploy()

        # After conversion, only layers up to eval_idx should remain
        assert len(decoder.layers) <= original_num_layers
        assert hasattr(decoder, "project")

    def test_value_op(self, decoder):
        """Test value operation for attention."""
        batch_size = 2
        seq_len = 100
        hidden_dim = 128
        spatial_shapes = [[10, 10]]

        memory = torch.randn(batch_size, seq_len, hidden_dim)

        values = decoder.value_op(
            memory=memory,
            value_proj=None,
            value_scale=None,
            memory_mask=None,
            memory_spatial_shapes=spatial_shapes,
        )

        assert isinstance(values, tuple)
        assert len(values) == len(spatial_shapes)
