# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmpretrain/blob/main/tests/test_models/test_backbones/test_vision_transformer.py
# https://github.com/open-mmlab/mmpretrain/blob/main/tests/test_models/test_backbones/utils.py
import math
from copy import deepcopy

import pytest
import torch
from otx.algo.classification.backbones import TimmVisionTransformer, VisionTransformer
from otx.algo.utils.mmengine_utils import load_checkpoint_to_model
from torch.nn import functional


def timm_resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    """Timm version pos embed resize function.

    copied from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = functional.interpolate(posemb_grid, size=gs_new, mode="bicubic", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


class TestVisionTransformer:
    @pytest.fixture()
    def config(self) -> dict:
        return {"arch": "b", "img_size": 224, "patch_size": 16, "drop_path_rate": 0.1}

    def test_structure(self, config):
        # Test invalid default arch
        cfg = deepcopy(config)
        cfg["arch"] = "unknown"
        with pytest.raises(ValueError, match="not in default archs"):
            VisionTransformer(**cfg)

        # Test invalid custom arch
        cfg = deepcopy(config)
        cfg["arch"] = {
            "num_layers": 24,
            "num_heads": 16,
            "feedforward_channels": 4096,
        }
        with pytest.raises(ValueError, match="Custom arch needs"):
            VisionTransformer(**cfg)

        # Test custom arch
        cfg = deepcopy(config)
        cfg["arch"] = {
            "embed_dims": 128,
            "num_layers": 24,
            "num_heads": 16,
            "feedforward_channels": 1024,
        }
        model = VisionTransformer(**cfg)
        assert model.embed_dims == 128
        assert model.num_layers == 24
        for layer in model.layers:
            assert layer.attn.num_heads == 16
            assert layer.ffn.feedforward_channels == 1024

        # Test out_indices
        cfg = deepcopy(config)
        cfg["out_indices"] = {1: 1}
        with pytest.raises(TypeError, match="get <class 'dict'>"):
            VisionTransformer(**cfg)
        cfg["out_indices"] = [0, 13]
        with pytest.raises(AssertionError, match="Invalid out_indices 13"):
            VisionTransformer(**cfg)

        # Test model structure
        cfg = deepcopy(config)
        model = VisionTransformer(**cfg)
        assert len(model.layers) == 12
        for layer in model.layers:
            assert layer.attn.embed_dims == 768
            assert layer.attn.num_heads == 12
            assert layer.ffn.feedforward_channels == 3072

        # Test model structure:  prenorm
        cfg = deepcopy(config)
        cfg["pre_norm"] = True
        model = VisionTransformer(**cfg)
        assert model.pre_norm.__class__ != torch.nn.Identity

    def test_init_weights(self, tmp_path, config):
        # test weight init cfg
        cfg = deepcopy(config)
        cfg["init_cfg"] = [
            {"type": "Kaiming", "layer": "Conv2d", "mode": "fan_in", "nonlinearity": "linear"},
        ]
        model = VisionTransformer(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()
        # The pos_embed is all zero before initialize
        assert torch.allclose(model.pos_embed, torch.tensor(0.0))

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        assert not torch.allclose(ori_weight, initialized_weight)
        assert not torch.allclose(model.pos_embed, torch.tensor(0.0))

        # test load checkpoint
        pretrain_pos_embed = model.pos_embed.clone().detach()
        checkpoint = tmp_path / "test.pth"
        torch.save(model.state_dict(), str(checkpoint))

        cfg = deepcopy(config)
        model = VisionTransformer(**cfg)
        state_dict = torch.load(str(checkpoint), None)
        load_checkpoint_to_model(model, state_dict, strict=True)
        assert torch.allclose(model.pos_embed, pretrain_pos_embed)

        # test load checkpoint with different img_size
        cfg = deepcopy(config)
        cfg["img_size"] = 384
        model = VisionTransformer(**cfg)
        state_dict = torch.load(str(checkpoint), None)
        load_checkpoint_to_model(model, state_dict, strict=True)
        resized_pos_embed = timm_resize_pos_embed(pretrain_pos_embed, model.pos_embed)
        assert torch.allclose(model.pos_embed, resized_pos_embed)

        checkpoint.unlink()

    def test_forward(self, config):
        imgs = torch.randn(1, 3, 224, 224)

        # test with_cls_token=False
        cfg = deepcopy(config)
        cfg["with_cls_token"] = False
        cfg["out_type"] = "cls_token"
        with pytest.raises(ValueError, match="must be True"):
            VisionTransformer(**cfg)

        cfg = deepcopy(config)
        cfg["with_cls_token"] = False
        cfg["out_type"] = "featmap"
        model = VisionTransformer(**cfg)
        outs = model(imgs)
        assert isinstance(outs, tuple)
        assert len(outs) == 1
        patch_token = outs[-1]
        assert patch_token.shape == (1, 768, 14, 14)

        # test with output cls_token
        cfg = deepcopy(config)
        model = VisionTransformer(**cfg)
        outs = model(imgs)
        assert isinstance(outs, tuple)
        assert len(outs) == 1
        cls_token = outs[-1]
        assert cls_token.shape == (1, 768)

        # Test forward with multi out indices
        cfg = deepcopy(config)
        cfg["out_indices"] = [-3, -2, -1]
        model = VisionTransformer(**cfg)
        outs = model(imgs)
        assert isinstance(outs, tuple)
        assert len(outs) == 3
        for out in outs:
            assert out.shape == (1, 768)

        # Test forward with dynamic input size
        imgs1 = torch.randn(1, 3, 224, 224)
        imgs2 = torch.randn(1, 3, 256, 256)
        imgs3 = torch.randn(1, 3, 256, 309)
        cfg = deepcopy(config)
        cfg["out_type"] = "featmap"
        model = VisionTransformer(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            assert isinstance(outs, tuple)
            assert len(outs) == 1
            patch_token = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 16), math.ceil(imgs.shape[3] / 16))
            assert patch_token.shape == (1, 768, *expect_feat_shape)

    def test_frozen_stages(self, config):
        cfg = deepcopy(config)
        cfg["arch"] = {
            "embed_dims": 128,
            "num_layers": 24,
            "num_heads": 16,
            "feedforward_channels": 1024,
        }
        cfg["frozen_stages"] = 1
        model = VisionTransformer(**cfg)
        # freeze position embedding
        if model.pos_embed is not None:
            assert not model.pos_embed.requires_grad
        # freeze patch embedding
        for param in model.patch_embed.parameters():
            assert not param.requires_grad
        # freeze pre-norm
        for param in model.pre_norm.parameters():
            assert not param.requires_grad
        # freeze cls_token
        if model.cls_token is not None:
            assert not model.cls_token.requires_grad
        # freeze layers
        for i in range(1, model.frozen_stages + 1):
            m = model.layers[i - 1]
            for param in m.parameters():
                assert not param.requires_grad

    def test_get_layer_depth(self, config):
        cfg = deepcopy(config)
        cfg["arch"] = {
            "embed_dims": 128,
            "num_layers": 24,
            "num_heads": 16,
            "feedforward_channels": 1024,
        }
        model = VisionTransformer(**cfg)
        layer_depth, num_layers = model.get_layer_depth("patch_embed")
        assert layer_depth == 0
        assert num_layers == model.num_layers + 2

        layer_depth, num_layers = model.get_layer_depth("pos_embed")
        assert layer_depth == 0
        assert num_layers == model.num_layers + 2

        layer_depth, num_layers = model.get_layer_depth("layers.1")
        assert layer_depth == 2
        assert num_layers == model.num_layers + 2


class TestTimmVisionTransformer:
    @pytest.fixture()
    def config(self) -> dict:
        return {"img_size": 224, "patch_size": 16, "drop_path_rate": 0.1}

    def test_init_weights(self, tmp_path, config):
        # test weight init cfg
        cfg = deepcopy(config)
        model = TimmVisionTransformer(**cfg)
        ori_weight = model.patch_embed.proj.weight.clone().detach()
        # The pos_embed is all zero before initialize
        assert torch.allclose(model.pos_embed, torch.tensor(0.0))

        model.init_weights()
        initialized_weight = model.patch_embed.proj.weight
        assert torch.allclose(ori_weight, initialized_weight)
        assert not torch.allclose(model.pos_embed, torch.tensor(0.0))

        # test load checkpoint
        pretrain_pos_embed = model.pos_embed.clone().detach()
        checkpoint = tmp_path / "test.pth"
        torch.save(model.state_dict(), str(checkpoint))

        cfg = deepcopy(config)
        model = TimmVisionTransformer(**cfg)
        state_dict = torch.load(str(checkpoint), None)
        load_checkpoint_to_model(model, state_dict, strict=True)
        assert torch.allclose(model.pos_embed, pretrain_pos_embed)

        checkpoint.unlink()

    def test_forward(self, config):
        imgs = torch.randn(1, 3, 224, 224)

        # test with output cls_token
        cfg = deepcopy(config)
        model = TimmVisionTransformer(**cfg)
        outs = model(imgs)
        assert isinstance(outs, tuple)
        assert len(outs) == 1
        cls_token = outs[-1]
        assert cls_token.shape == (1, 768)

        # test forward output raw
        outs = model(imgs, out_type="raw")
        assert isinstance(outs, tuple)
        assert len(outs) == 1
        feat = outs[-1]
        assert feat.shape == (1, 197, 768)
