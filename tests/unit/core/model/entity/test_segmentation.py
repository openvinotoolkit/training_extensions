# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for segmentation model entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from omegaconf import OmegaConf
from otx.core.model.entity.segmentation import MMSegCompatibleModel

if TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig


class TestOTXSegmentationModel:
    @pytest.fixture()
    def config(self) -> DictConfig:
        return OmegaConf.load("src/otx/recipe/segmentation/segnext_s.yaml")

    @pytest.fixture()
    def model(self, config) -> MMSegCompatibleModel:
        return MMSegCompatibleModel(config.model.otx_model.config)

    def test_create_model(self, model) -> None:
        mmseg_model = model._create_model()
        assert mmseg_model is not None
        assert isinstance(mmseg_model, torch.nn.Module)

    def test_customize_inputs(self, model, fxt_seg_data_entity) -> None:
        output_data = model._customize_inputs(fxt_seg_data_entity[2])
        assert output_data is not None
        assert output_data["data_samples"][-1].metainfo["pad_shape"] == output_data["inputs"].shape[-2:]
        assert (
            output_data["data_samples"][-1].metainfo["pad_shape"]
            == output_data["data_samples"][-1].gt_sem_seg.data.shape[-2:]
        )

    def test_customize_outputs(self, model, fxt_seg_data_entity) -> None:
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample
        from otx.core.data.entity.base import OTXBatchLossEntity
        from otx.core.data.entity.segmentation import SegBatchPredEntity

        data_sample = SegDataSample()
        pred_segm_map = PixelData()
        pred_segm_map.data = torch.randint(0, 2, (1, 4, 4))
        data_sample.pred_sem_seg = pred_segm_map

        output_loss = {"loss_segm": torch.rand(1, requires_grad=True), "acc": torch.rand(1), "some": "some"}
        out = model._customize_outputs(output_loss, fxt_seg_data_entity[2])
        assert isinstance(out, OTXBatchLossEntity)

        model.training = False
        out = model._customize_outputs([data_sample], fxt_seg_data_entity[2])
        assert isinstance(out, SegBatchPredEntity)
