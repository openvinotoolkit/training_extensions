# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmpretrain/blob/main/tests/test_models/test_backbones/test_vision_transformer.py
# https://github.com/open-mmlab/mmpretrain/blob/main/tests/test_models/test_backbones/utils.py
from copy import deepcopy

import pytest
import torch
from otx.algo.classification.backbones import VisionTransformer
from otx.algo.utils.mmengine_utils import load_checkpoint_to_model


class TestVisionTransformer:
    @pytest.fixture()
    def config(self) -> dict:
        return {"arch": "vit-tiny", "img_size": 224, "patch_size": 16, "drop_path_rate": 0.1}

    def test_init_weights(self, tmp_path, config):
        # test weight init cfg
        cfg = deepcopy(config)
        model = VisionTransformer(**cfg)
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
        model = VisionTransformer(**cfg)
        state_dict = torch.load(str(checkpoint), None)
        load_checkpoint_to_model(model, state_dict, strict=True)
        assert torch.allclose(model.pos_embed, pretrain_pos_embed)

        checkpoint.unlink()

    def test_forward(self, config):
        imgs = torch.randn(1, 3, 224, 224)

        # test with output cls_token
        cfg = deepcopy(config)
        model = VisionTransformer(**cfg)
        outs = model(imgs)
        assert isinstance(outs, tuple)
        assert len(outs) == 1
        cls_token = outs[-1]
        assert cls_token.shape == (1, model.embed_dim)

        # test forward output raw
        outs = model(imgs, out_type="raw")
        assert isinstance(outs, tuple)
        assert len(outs) == 1
        feat = outs[-1]
        assert feat.shape == (1, 197, model.embed_dim)
