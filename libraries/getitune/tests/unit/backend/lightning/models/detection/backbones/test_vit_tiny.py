# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Vision Transformer (ViT) Tiny backbone."""

import pytest
import torch

from getitune.backend.lightning.models.detection.backbones.vit_tiny import (
    Attention,
    Block,
    DropPath,
    SimplifiedPatchEmbed,
    VisionTransformer,
    apply_rope,
    drop_path,
    rotate_half,
)
from getitune.backend.lightning.models.modules.drop import DropPath as SharedDropPath


class TestHelperFunctions:
    """Test helper functions for ViT."""

    def test_rotate_half(self):
        """Test rotate_half function."""
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
        result = rotate_half(x)

        # First half becomes negated second half
        expected = torch.tensor([[-3, -4, 1, 2], [-7, -8, 5, 6]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_rotate_half_batched(self):
        """Test rotate_half with batched input."""
        x = torch.randn(2, 4, 8, 64)  # (B, H, N, D)
        result = rotate_half(x)

        assert result.shape == x.shape
        # Check the rotation pattern
        assert torch.allclose(result[..., :32], -x[..., 32:])
        assert torch.allclose(result[..., 32:], x[..., :32])

    def test_apply_rope(self):
        """Test apply_rope function."""
        x = torch.randn(2, 4, 196, 64)
        sin = torch.randn(1, 1, 196, 64)
        cos = torch.randn(1, 1, 196, 64)

        result = apply_rope(x, sin, cos)

        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_drop_path_no_drop(self):
        """Test drop_path with 0 probability."""
        x = torch.randn(2, 100, 256)
        result = drop_path(x, drop_prob=0.0, training=True)
        assert torch.allclose(result, x)

    def test_drop_path_eval_mode(self):
        """Test drop_path in eval mode (training=False)."""
        x = torch.randn(2, 100, 256)
        result = drop_path(x, drop_prob=0.5, training=False)
        assert torch.allclose(result, x)

    def test_drop_path_training(self):
        """Test drop_path during training."""
        torch.manual_seed(42)
        x = torch.ones(10, 100, 256)
        result = drop_path(x, drop_prob=0.5, training=True)

        # Some samples may be zeroed out, others scaled
        # Check that the shape is preserved
        assert result.shape == x.shape


class TestDropPath:
    """Test DropPath module."""

    def test_droppath_init(self):
        """Test DropPath initialization."""
        dp = DropPath(drop_prob=0.1)
        assert dp.drop_prob == 0.1

    def test_droppath_init_none(self):
        """Test DropPath with None probability."""
        dp = DropPath(drop_prob=None)
        assert dp.drop_prob is None

    def test_droppath_forward_training(self):
        """Test DropPath forward in training mode."""
        dp = DropPath(drop_prob=0.5)
        dp.train()
        x = torch.randn(4, 100, 256)
        result = dp(x)
        assert result.shape == x.shape

    def test_droppath_forward_eval(self):
        """Test DropPath forward in eval mode."""
        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(4, 100, 256)
        result = dp(x)
        assert torch.allclose(result, x)


class TestSimplifiedPatchEmbed:
    """Test SimplifiedPatchEmbed module."""

    @pytest.fixture
    def patch_embed(self):
        """Create patch embed layer."""
        return SimplifiedPatchEmbed(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=192,
        )

    def test_init(self, patch_embed):
        """Test patch embed initialization."""
        assert patch_embed.grid_size == (14, 14)
        assert patch_embed.num_patches == 196
        assert patch_embed.proj.kernel_size == (16, 16)
        assert patch_embed.proj.stride == (16, 16)

    def test_init_tuple_sizes(self):
        """Test patch embed with tuple sizes."""
        pe = SimplifiedPatchEmbed(
            img_size=(224, 224),
            patch_size=(16, 16),
            in_chans=3,
            embed_dim=192,
        )
        assert pe.grid_size == (14, 14)

    def test_forward(self, patch_embed):
        """Test patch embed forward pass."""
        x = torch.randn(2, 3, 224, 224)
        output = patch_embed(x)

        # Output shape: (B, num_patches, embed_dim)
        assert output.shape == (2, 196, 192)

    def test_forward_different_sizes(self):
        """Test patch embed with different image sizes."""
        pe = SimplifiedPatchEmbed(
            img_size=448,
            patch_size=16,
            in_chans=3,
            embed_dim=256,
        )
        x = torch.randn(1, 3, 448, 448)
        output = pe(x)

        # 448/16 = 28, so 28*28 = 784 patches
        assert output.shape == (1, 784, 256)


class TestAttention:
    """Test Attention module."""

    @pytest.fixture
    def attention(self):
        """Create attention module."""
        return Attention(
            dim=192,
            num_heads=3,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
        )

    def test_init(self, attention):
        """Test attention initialization."""
        assert attention.num_heads == 3
        assert attention.scale == (192 // 3) ** -0.5
        assert attention.qkv.in_features == 192
        assert attention.qkv.out_features == 192 * 3
        assert attention.proj.in_features == 192
        assert attention.proj.out_features == 192

    def test_forward_without_rope(self, attention):
        """Test attention forward without RoPE."""
        x = torch.randn(2, 197, 192)  # (B, N, C) with cls token
        output = attention(x)

        assert output.shape == x.shape

    def test_forward_with_rope(self, attention):
        """Test attention forward with RoPE."""
        x = torch.randn(2, 197, 192)  # (B, N, C) with cls token

        # Create RoPE sin/cos for patches (excluding cls token)
        sin = torch.randn(1, 1, 196, 64)  # 192 / 3 = 64 head dim
        cos = torch.randn(1, 1, 196, 64)

        output = attention(x, rope_sincos=(sin, cos))

        assert output.shape == x.shape

    def test_forward_batched(self, attention):
        """Test attention with different batch sizes."""
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 197, 192)
            output = attention(x)
            assert output.shape == x.shape


class TestBlock:
    """Test Transformer Block module."""

    @pytest.fixture
    def block(self):
        """Create transformer block."""
        return Block(
            dim=192,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            attn_drop=0.0,
            drop_path=0.0,
            drop=0.0,
        )

    @pytest.fixture
    def block_with_droppath(self):
        """Create transformer block with drop path."""
        return Block(
            dim=192,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=0.1,
        )

    def test_init(self, block):
        """Test block initialization."""
        assert hasattr(block, "norm1")
        assert hasattr(block, "attn")
        assert hasattr(block, "norm2")
        assert hasattr(block, "mlp")
        assert hasattr(block, "drop_path")

    def test_init_with_droppath(self, block_with_droppath):
        """Test block with drop path initialization."""
        # `Block` is now shared (common/layers/vit_blocks.py) and uses the canonical
        # `modules.drop.DropPath` internally, not vit_tiny's own standalone `DropPath`
        # (the two are behaviorally identical; `vit_tiny.DropPath` is kept for its own
        # direct unit tests below and is no longer used internally by `Block`).
        assert isinstance(block_with_droppath.drop_path, SharedDropPath)
        assert block_with_droppath.drop_path.drop_prob == 0.1

    def test_forward_without_rope(self, block):
        """Test block forward without RoPE."""
        x = torch.randn(2, 197, 192)
        output = block(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_with_rope(self, block):
        """Test block forward with RoPE."""
        x = torch.randn(2, 197, 192)
        sin = torch.randn(1, 1, 196, 64)
        cos = torch.randn(1, 1, 196, 64)

        output = block(x, rope_sincos=(sin, cos))

        assert output.shape == x.shape


class TestVisionTransformer:
    """Test VisionTransformer module."""

    @pytest.fixture
    def vit(self):
        """Create ViT model."""
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            return_layers=[3, 7, 11],
        )

    @pytest.fixture
    def vit_small(self):
        """Create smaller ViT for faster testing."""
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=128,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            return_layers=[1, 2, 3],
        )

    def test_init(self, vit):
        """Test ViT initialization."""
        assert vit.num_features == 192
        assert vit.embed_dim == 192
        assert vit.num_tokens == 1
        assert vit.return_layers == [3, 7, 11]
        assert vit.patch_size == 16

    def test_init_components(self, vit):
        """Test ViT components initialization."""
        model = vit.get_model()

        assert hasattr(model, "patch_embed")
        assert hasattr(model, "cls_token")
        assert hasattr(model, "blocks")
        assert hasattr(model, "rope_embed")

        assert len(model.blocks) == 12
        assert model.cls_token.shape == (1, 1, 192)

    def test_forward(self, vit_small):
        """Test ViT forward pass."""
        x = torch.randn(2, 3, 224, 224)
        outputs = vit_small(x)

        assert len(outputs) == 3  # 3 return layers
        for patch_features, cls_token in outputs:
            assert patch_features.shape == (2, 196, 128)
            assert cls_token.shape == (2, 128)

    def test_forward_different_batch_sizes(self, vit_small):
        """Test ViT with different batch sizes."""
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 224, 224)
            outputs = vit_small(x)

            assert len(outputs) == 3
            for patch_features, cls_token in outputs:
                assert patch_features.shape[0] == batch_size
                assert cls_token.shape[0] == batch_size

    def test_forward_different_image_sizes(self, vit_small):
        """Test ViT with different image sizes."""
        # ViT can handle different input sizes
        for img_size in [224, 448, 640]:
            x = torch.randn(1, 3, img_size, img_size)
            outputs = vit_small(x)

            num_patches = (img_size // 16) ** 2
            for patch_features, cls_token in outputs:
                assert patch_features.shape == (1, num_patches, 128)
                assert cls_token.shape == (1, 128)

    def test_feature_dim(self, vit_small):
        """Test feature_dim method."""
        assert vit_small.feature_dim() == 128

    def test_get_model(self, vit_small):
        """Test get_model method."""
        model = vit_small.get_model()
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "blocks")

    def test_no_weight_decay(self, vit_small):
        """Test no_weight_decay method."""
        no_wd = vit_small.no_weight_decay()
        assert "cls_token" in no_wd

    def test_init_weights(self, vit_small):
        """Test weight initialization."""
        # Simply verify init_weights runs without error
        vit_small.init_weights()

        model = vit_small.get_model()
        # Check cls_token is not all zeros after init
        assert not torch.all(model.cls_token == 0)

    def test_custom_return_layers(self):
        """Test ViT with custom return layers."""
        vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=128,
            depth=6,
            num_heads=4,
            return_layers=[0, 2, 5],
        )

        x = torch.randn(1, 3, 224, 224)
        outputs = vit(x)

        assert len(outputs) == 3

    def test_default_return_layers(self):
        """Test ViT with default return layers."""
        vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=128,
            depth=12,
            num_heads=4,
            # return_layers defaults to [3, 7, 11]
        )

        assert vit.return_layers == [3, 7, 11]

    def test_with_drop_path(self):
        """Test ViT with drop path rate."""
        vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=128,
            depth=4,
            num_heads=4,
            drop_path_rate=0.1,
            return_layers=[1, 2, 3],
        )

        x = torch.randn(1, 3, 224, 224)
        outputs = vit(x)

        assert len(outputs) == 3

    def test_with_dropout(self):
        """Test ViT with dropout."""
        vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=128,
            depth=4,
            num_heads=4,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            return_layers=[1, 2, 3],
        )

        x = torch.randn(1, 3, 224, 224)
        outputs = vit(x)

        assert len(outputs) == 3


class TestGradientFlow:
    """Test gradient flow through ViT."""

    @pytest.fixture
    def vit(self):
        """Create ViT for gradient testing."""
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=128,
            depth=4,
            num_heads=4,
            return_layers=[1, 2, 3],
        )

    def test_gradient_flow(self, vit):
        """Test that gradients flow through the model."""
        vit.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)

        outputs = vit(x)

        # Sum all outputs for loss
        loss = sum(patch_feat.sum() + cls_token.sum() for patch_feat, cls_token in outputs)
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_to_parameters(self, vit):
        """Test that gradients flow to parameters."""
        vit.train()
        x = torch.randn(2, 3, 224, 224)

        outputs = vit(x)
        loss = sum(patch_feat.sum() + cls_token.sum() for patch_feat, cls_token in outputs)
        loss.backward()

        model = vit.get_model()

        # Check key parameters have gradients
        assert model.cls_token.grad is not None
        assert model.blocks[0].attn.qkv.weight.grad is not None
