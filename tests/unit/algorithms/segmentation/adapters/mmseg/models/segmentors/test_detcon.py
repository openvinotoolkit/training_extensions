# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from copy import deepcopy
from functools import partial, reduce
from typing import List, Union

import pytest
import torch
import torch.nn as nn

from otx.algorithms.segmentation.adapters.mmseg import DetConB, SupConDetConB
from otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon import (
    MaskPooling,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@pytest.fixture(autouse=True)
def setup_module(monkeypatch, mocker):
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pipeline1 = nn.Sequential(nn.Conv2d(3, 1, (1, 1), bias=False), nn.Conv2d(1, 1, (1, 1), bias=False))
            self.pipeline2 = nn.Sequential(
                nn.Conv2d(3, 1, (1, 1), stride=2, bias=False), nn.Conv2d(1, 1, (1, 1), bias=False)
            )

        def init_weights(self, init_linear=None):
            pass

        def forward(self, x):
            return [self.pipeline1(x), self.pipeline2(x)]

    class MockNeck(nn.Sequential):
        def __init__(self):
            super().__init__(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))

        def init_weights(self, init_linear=None):
            pass

    class MockDecodeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.align_corners = None
            self.num_classes = 1
            self.out_channels = 1

        def forward(self, x):
            return x

    def build_mock(mock_class, *args, **kwargs):
        return mock_class()

    monkeypatch.setattr(
        "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon.build_backbone",
        partial(build_mock, MockBackbone),
    )
    monkeypatch.setattr(
        "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon.build_neck", partial(build_mock, MockNeck)
    )
    monkeypatch.setattr(
        "mmseg.models.segmentors.encoder_decoder.builder.build_backbone", partial(build_mock, MockBackbone)
    )
    monkeypatch.setattr(
        "mmseg.models.segmentors.encoder_decoder.builder.build_head", partial(build_mock, MockDecodeHead)
    )
    mocker.patch(
        "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon.build_loss",
        return_value=lambda *args, **kwargs: dict(loss=1.0),
    )
    mocker.patch(
        "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon.DetConB._register_state_dict_hook"
    )
    mocker.patch("otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon.DetConB.state_dict_hook")
    mocker.patch("otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon.load_checkpoint")
    mocker.patch(
        "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.detcon.SupConDetConB._decode_head_forward_train",
        return_value=(dict(loss=1.0), None),
    )


class TestMaskPooling:
    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "masks", [torch.Tensor([[[[0, 1], [1, 2]]]]).to(torch.int64), torch.Tensor([[[0, 1], [1, 2]]]).to(torch.int64)]
    )
    @pytest.mark.parametrize("num_classes", [256, 3])
    def test_pool_masks(self, masks, num_classes):
        """Test pool_masks function."""
        mask_pooling = MaskPooling(num_classes=num_classes, downsample=1)

        binary_masks = mask_pooling.pool_masks(masks)

        assert binary_masks.shape[1] == num_classes

    @e2e_pytest_unit
    def test_sample_masks(self):
        """Test sample_masks function."""
        num_classes = 3
        masks = torch.Tensor([[[[0, 1], [1, 2]]]]).to(torch.int64)
        mask_pooling = MaskPooling(num_classes=num_classes, downsample=1)

        binary_masks = mask_pooling.pool_masks(masks)
        _, sampled_mask_ids = mask_pooling.sample_masks(binary_masks)

        assert len(torch.unique(sampled_mask_ids)) > 1
        assert len(torch.unique(sampled_mask_ids)) <= num_classes

    @e2e_pytest_unit
    def test_forward(self):
        """Test forward function."""
        num_classes = 3
        masks = torch.Tensor([[[[0, 1], [1, 2]]]]).to(torch.int64)
        mask_pooling = MaskPooling(num_classes=num_classes, downsample=1)

        sampled_masks, _ = mask_pooling(masks)

        assert torch.all(sampled_masks.sum(dim=-1) == 1)


class TestDetConB:
    """Test DetConB."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.detconb = DetConB(backbone={}, neck={}, head={}, downsample=1, loss_cfg=dict(type="DetConLoss"))

    @e2e_pytest_unit
    def test_init_weights(self) -> None:
        """Test init_weights function."""
        for param_ol, param_tgt in zip(
            self.detconb.online_backbone.parameters(), self.detconb.target_backbone.parameters()
        ):
            assert torch.all(param_ol == param_tgt)
            assert param_ol.requires_grad
            assert not param_tgt.requires_grad

        for param_ol, param_tgt in zip(
            self.detconb.online_projector.parameters(), self.detconb.target_projector.parameters()
        ):
            assert torch.all(param_ol == param_tgt)
            assert param_ol.requires_grad
            assert not param_tgt.requires_grad

    @e2e_pytest_unit
    def test_momentum_update(self) -> None:
        """Test _momentum_update function."""
        original_params = {"backbone": [], "projector": []}
        for param_tgt in self.detconb.target_backbone.parameters():
            param_tgt.data *= 2.0
            original_params["backbone"].append(deepcopy(param_tgt))

        for param_tgt in self.detconb.target_projector.parameters():
            param_tgt.data *= 2.0
            original_params["projector"].append(deepcopy(param_tgt))

        self.detconb._momentum_update()

        for param_ol, param_tgt, orig_tgt in zip(
            self.detconb.online_backbone.parameters(),
            self.detconb.target_backbone.parameters(),
            original_params["backbone"],
        ):
            assert torch.all(
                param_tgt.data == orig_tgt * self.detconb.momentum + param_ol.data * (1.0 - self.detconb.momentum)
            )

        for param_ol, param_tgt, orig_tgt in zip(
            self.detconb.online_projector.parameters(),
            self.detconb.target_projector.parameters(),
            original_params["projector"],
        ):
            assert torch.all(
                param_tgt.data == orig_tgt * self.detconb.momentum + param_ol.data * (1.0 - self.detconb.momentum)
            )

    @e2e_pytest_unit
    @pytest.mark.parametrize("input_transform", ["resize_concat", "multiple_select", "etc"])
    @pytest.mark.parametrize(
        "inputs,in_index",
        [
            ([torch.ones(1, 1, 4, 4), torch.ones(1, 1, 2, 2)], [0, 1]),
            ([torch.ones(1, 1, 4, 4), torch.ones(1, 1, 2, 2)], [0]),
            ([torch.ones(1, 1, 4, 4)], [0]),
        ],
    )
    def test_transform_inputs(self, inputs: List, in_index: Union[List, int], input_transform: str) -> None:
        """Test transform_inputs function."""
        setattr(self.detconb, "in_index", in_index)
        setattr(self.detconb, "input_transform", input_transform)

        transformed_inputs = self.detconb.transform_inputs(inputs)

        if input_transform == "resize_concat":
            assert transformed_inputs.shape[1] == len(in_index)
        elif input_transform == "multiple_select":
            assert len(transformed_inputs) == len(in_index)
        else:
            assert len(transformed_inputs) == 1

    @e2e_pytest_unit
    def test_sample_masked_feats(self) -> None:
        """Test sample_masked_feats function."""
        setattr(self.detconb, "in_index", [0, 1])
        setattr(self.detconb, "input_transform", "resize_concat")
        feats = [torch.randn((1, 1, 4, 4)), torch.randn((1, 1, 2, 2))]
        masks = torch.randint(0, 3, (1, 1, 4, 4), dtype=torch.int64)

        proj, sampled_mask_ids = self.detconb.sample_masked_feats(feats, masks, self.detconb.online_projector)

        assert proj.ndim == 2
        assert proj.shape[0] == reduce(lambda x, y: x * y, feats[0].shape[2:])
        assert proj.shape[1] == sum([feat.shape[1] for feat in feats])
        assert proj.shape[0] == sampled_mask_ids.shape[1]

    @e2e_pytest_unit
    def test_detconb_train_step(self) -> None:
        """Test train_step function wraps forward and _parse_losses."""
        setattr(self.detconb, "in_index", [0, 1])
        setattr(self.detconb, "input_transform", "resize_concat")
        img = torch.randn((1, 2, 3, 4, 4))
        gt_semantic_seg = torch.randint(0, 3, (1, 1, 2, 4, 4), dtype=torch.int64)

        outputs = self.detconb.train_step(
            data_batch=dict(img=img, img_metas={}, gt_semantic_seg=gt_semantic_seg), optimizer=None
        )

        assert "loss" in outputs
        assert "log_vars" in outputs
        assert "num_samples" in outputs


class TestSupConDetConB:
    """Test SupConDetConB."""

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "img,gt_semantic_seg,expected",
        [
            (torch.ones((1, 2, 3, 4, 4), dtype=torch.float32), torch.ones((1, 1, 2, 4, 4), dtype=torch.int64), True),
            (torch.ones((1, 3, 4, 4), dtype=torch.float32), torch.ones((1, 1, 4, 4), dtype=torch.int64), False),
        ],
    )
    def test_forward_train(self, img: torch.Tensor, gt_semantic_seg: torch.Tensor, expected: bool):
        """Test forward_train function."""
        supcon_detconb = SupConDetConB(
            backbone={},
            neck={},
            head={},
            decode_head={},
            downsample=1,
            input_transform="resize_concat",
            in_index=[0, 1],
            loss_cfg=dict(type="DetConLoss"),
            task_adapt=dict(dst_classes=1, src_classes=1),
        )

        results = supcon_detconb(img=img, img_metas=[], gt_semantic_seg=gt_semantic_seg)

        assert ("loss_detcon" in results) == expected
