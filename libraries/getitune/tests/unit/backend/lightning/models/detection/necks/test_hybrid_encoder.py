# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of HybridEncoder."""

import pytest
import torch

from getitune.backend.lightning.models.detection.necks.hybrid_encoder import HybridEncoder, HybridEncoderModule


class TestHybridEncoderModule:
    """Test class for HybridEncoderModule."""

    @pytest.fixture
    def encoder_default(self):
        """Create encoder with default settings."""
        return HybridEncoderModule(
            in_channels=[512, 1024, 2048],
            hidden_dim=256,
            feat_strides=[8, 16, 32],
        )

    @pytest.fixture
    def encoder_small(self):
        """Create a smaller encoder for faster testing."""
        return HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            feat_strides=[8, 16, 32],
            dim_feedforward=512,
            num_encoder_layers=1,
        )

    @pytest.fixture
    def encoder_with_eval_size(self):
        """Create encoder with evaluation spatial size."""
        return HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            feat_strides=[8, 16, 32],
            eval_spatial_size=(640, 640),
        )

    def test_init_default(self, encoder_default):
        """Test default initialization."""
        assert isinstance(encoder_default, HybridEncoderModule)
        assert encoder_default.hidden_dim == 256
        assert encoder_default.in_channels == [512, 1024, 2048]
        assert encoder_default.feat_strides == [8, 16, 32]
        assert encoder_default.use_encoder_idx == [2]
        assert encoder_default.num_encoder_layers == 1

    def test_init_components(self, encoder_small):
        """Test that all components are initialized."""
        # Input projection
        assert hasattr(encoder_small, "input_proj")
        assert len(encoder_small.input_proj) == 3

        # Encoder
        assert hasattr(encoder_small, "encoder")
        assert len(encoder_small.encoder) == 1  # Only index 2 by default

        # FPN components
        assert hasattr(encoder_small, "lateral_convs")
        assert hasattr(encoder_small, "fpn_blocks")
        assert len(encoder_small.lateral_convs) == 2
        assert len(encoder_small.fpn_blocks) == 2

        # PAN components
        assert hasattr(encoder_small, "downsample_convs")
        assert hasattr(encoder_small, "pan_blocks")
        assert len(encoder_small.downsample_convs) == 2
        assert len(encoder_small.pan_blocks) == 2

    def test_out_channels(self, encoder_small):
        """Test output channels property."""
        assert encoder_small.out_channels == [128, 128, 128]
        assert encoder_small.out_strides == [8, 16, 32]

    def test_forward(self, encoder_small):
        """Test forward pass."""
        batch_size = 2
        feats = [
            torch.randn(batch_size, 128, 80, 80),
            torch.randn(batch_size, 256, 40, 40),
            torch.randn(batch_size, 512, 20, 20),
        ]

        outputs = encoder_small(feats)

        assert len(outputs) == 3
        assert outputs[0].shape == (batch_size, 128, 80, 80)
        assert outputs[1].shape == (batch_size, 128, 40, 40)
        assert outputs[2].shape == (batch_size, 128, 20, 20)

    def test_forward_different_batch_sizes(self, encoder_small):
        """Test forward with different batch sizes."""
        for batch_size in [1, 2, 4]:
            feats = [
                torch.randn(batch_size, 128, 80, 80),
                torch.randn(batch_size, 256, 40, 40),
                torch.randn(batch_size, 512, 20, 20),
            ]

            outputs = encoder_small(feats)

            assert len(outputs) == 3
            assert outputs[0].shape[0] == batch_size

    def test_forward_mismatched_features_raises(self, encoder_small):
        """Test that mismatched feature count raises error."""
        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            # Missing third feature
        ]

        with pytest.raises(ValueError, match="Input feature size"):
            encoder_small(feats)

    def test_forward_training_mode(self, encoder_small):
        """Test forward in training mode."""
        encoder_small.train()
        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        outputs = encoder_small(feats)
        assert len(outputs) == 3

    def test_forward_eval_mode(self, encoder_small):
        """Test forward in eval mode."""
        encoder_small.eval()
        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        with torch.no_grad():
            outputs = encoder_small(feats)

        assert len(outputs) == 3

    def test_init_weights_with_eval_size(self, encoder_with_eval_size):
        """Test weight initialization with eval spatial size."""
        encoder_with_eval_size.init_weights()

        # Check that position embedding is created for encoder index
        assert hasattr(encoder_with_eval_size, "pos_embed2")

    def test_build_2d_sincos_position_embedding(self):
        """Test 2D sin-cos position embedding generation."""
        w, h = 20, 20
        embed_dim = 256
        temperature = 10000.0

        pos_embed = HybridEncoderModule.build_2d_sincos_position_embedding(w, h, embed_dim, temperature)

        assert pos_embed.shape == (1, w * h, embed_dim)
        assert not torch.isnan(pos_embed).any()

    def test_build_2d_sincos_position_embedding_invalid_dim(self):
        """Test that invalid embed_dim raises error."""
        with pytest.raises(ValueError, match="divisible by 4"):
            HybridEncoderModule.build_2d_sincos_position_embedding(10, 10, 255, 10000.0)

    def test_no_encoder_layers(self):
        """Test encoder with no encoder layers."""
        encoder = HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            num_encoder_layers=0,
        )

        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        outputs = encoder(feats)
        assert len(outputs) == 3

    def test_multiple_encoder_indices(self):
        """Test encoder with multiple encoder indices."""
        encoder = HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            use_encoder_idx=[1, 2],
            num_encoder_layers=1,
        )

        assert len(encoder.encoder) == 2

        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        outputs = encoder(feats)
        assert len(outputs) == 3

    def test_custom_activation(self):
        """Test encoder with custom activation."""
        encoder = HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            activation=torch.nn.ReLU,
            enc_activation=torch.nn.ReLU,
        )

        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        outputs = encoder(feats)
        assert len(outputs) == 3

    def test_depth_mult(self):
        """Test encoder with depth multiplier."""
        encoder = HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            depth_mult=0.5,
        )

        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        outputs = encoder(feats)
        assert len(outputs) == 3

    def test_expansion(self):
        """Test encoder with expansion factor."""
        encoder = HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            expansion=0.5,
        )

        feats = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        outputs = encoder(feats)
        assert len(outputs) == 3


class TestHybridEncoderFactory:
    """Test class for HybridEncoder factory."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "rtdetr_18",
            "rtdetr_50",
            "rtdetr_101",
        ],
    )
    def test_factory_creates_module(self, model_name):
        """Test that factory creates HybridEncoderModule."""
        encoder = HybridEncoder(model_name)
        assert isinstance(encoder, HybridEncoderModule)

    def test_factory_rtdetr_18_config(self):
        """Test RTDETR-18 configuration."""
        encoder = HybridEncoder("rtdetr_18")
        assert encoder.in_channels == [128, 256, 512]

    def test_factory_rtdetr_50_config(self):
        """Test RTDETR-50 configuration (defaults)."""
        encoder = HybridEncoder("rtdetr_50")
        # Uses default values
        assert encoder.hidden_dim == 256

    def test_factory_rtdetr_101_config(self):
        """Test RTDETR-101 configuration."""
        encoder = HybridEncoder("rtdetr_101")
        assert encoder.hidden_dim == 384
        assert encoder.in_channels == [512, 1024, 2048]

    def test_factory_with_eval_size(self):
        """Test factory with evaluation spatial size."""
        encoder = HybridEncoder("rtdetr_18", eval_spatial_size=(640, 640))
        assert encoder.eval_spatial_size == (640, 640)

    def test_factory_invalid_model_raises(self):
        """Test that invalid model name raises error."""
        with pytest.raises(KeyError, match="not supported"):
            HybridEncoder("invalid_model")


class TestGradientFlow:
    """Test gradient flow through HybridEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for gradient testing."""
        return HybridEncoderModule(
            in_channels=[128, 256, 512],
            hidden_dim=128,
            num_encoder_layers=1,
        )

    def test_gradient_flow(self, encoder):
        """Test that gradients flow through the encoder."""
        encoder.train()
        feats = [
            torch.randn(2, 128, 40, 40, requires_grad=True),
            torch.randn(2, 256, 20, 20, requires_grad=True),
            torch.randn(2, 512, 10, 10, requires_grad=True),
        ]

        outputs = encoder(feats)
        loss = sum(out.sum() for out in outputs)
        loss.backward()

        # Check gradients exist for all inputs
        for feat in feats:
            assert feat.grad is not None
            assert not torch.all(feat.grad == 0)
