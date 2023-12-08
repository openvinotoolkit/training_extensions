# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import hydra
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from otx.cli.utils.hydra import configure_hydra_outputs
from otx.core.config import register_configs
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity


class TestOTXSegmentationModel:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        GlobalHydra.instance().clear()
        register_configs()
        initialize(config_path="../../../../../src/otx/config", version_base="1.3", job_name="otx_train")
        overrides_list = ['+recipe=segmentation/litehrnet_18.yaml', 'base.output_dir=/tmp/']
        self.cfg = compose(config_name="train", overrides=overrides_list, return_hydra_config=True)
        configure_hydra_outputs(self.cfg)
        self.mmseg_model = hydra.utils.instantiate(self.cfg.model.otx_model)
        img_size = (240,240)
        self.inputs = SegBatchDataEntity(1, [torch.rand(img_size)],
                                        [ImageInfo(0, img_size, img_size, img_size, img_size)],
                                        [torch.rand(img_size)])

    def test_create_model(self) -> None:
        model = self.mmseg_model._create_model()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_customize_inputs(self) -> None:
        output_data = self.mmseg_model._customize_inputs(self.inputs)
        assert output_data is not None
        assert output_data["data_samples"][-1].metainfo["pad_shape"] == output_data["inputs"].shape[-2:]
        assert output_data["data_samples"][-1].metainfo["pad_shape"] == output_data["data_samples"][-1].gt_sem_seg.data.shape[-2:]

    def test_customize_outputs(self) -> None:
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample

        from otx.core.data.entity.base import OTXBatchLossEntity
        from otx.core.data.entity.segmentation import SegBatchPredEntity

        data_sample = SegDataSample()
        pred_segm_map = PixelData()
        pred_segm_map.data = torch.randint(0, 2, (1, 4, 4))
        data_sample.pred_sem_seg = pred_segm_map

        output_loss = {"loss_segm": torch.rand(1, requires_grad=True),
                       "acc": torch.rand(1),
                       "some": "some"}
        out = self.mmseg_model._customize_outputs(output_loss, self.inputs)
        assert isinstance(out, OTXBatchLossEntity)

        self.mmseg_model.training = False
        out = self.mmseg_model._customize_outputs([data_sample], self.inputs)
        assert isinstance(out, SegBatchPredEntity)
