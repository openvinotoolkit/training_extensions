# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for segmentation model entity."""

from __future__ import annotations

import pytest
import torch
from otx.core.model.segmentation import MMSegCompatibleModel
from otx.core.types.label import SegLabelInfo


class TestOTXSegmentationModel:
    @pytest.fixture()
    def model(self, mocker) -> MMSegCompatibleModel:
        mocker.patch("otx.core.model.segmentation.inplace_num_classes")
        mocker.patch.object(MMSegCompatibleModel, "_create_model")
        model = MMSegCompatibleModel(label_info=3, config=mocker.MagicMock())
        model.label_info = SegLabelInfo(
            label_names=["Background", "label_0", "label_1"],
            label_groups=[["Background", "label_0", "label_1"]],
        )
        return model

    def test_customize_inputs(self, model, fxt_seg_data_entity, mocker) -> None:
        model.model.data_preprocessor = mocker.MagicMock()
        output_data = model._customize_inputs(fxt_seg_data_entity[2])
        assert output_data is not None
        assert isinstance(output_data, mocker.MagicMock)

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

    def test_validation_step(self, mocker, model, fxt_seg_data_entity) -> None:
        model.eval()
        model.on_validation_start()
        mocker_update_loss = mocker.patch.object(
            model,
            "_convert_pred_entity_to_compute_metric",
            return_value=[
                {"preds": torch.randint(0, 2, size=[1, 3, 3]), "target": torch.randint(0, 2, size=[1, 3, 3])},
            ],
        )
        model.validation_step(fxt_seg_data_entity[2], 0)
        mocker_update_loss.assert_called_once()

    def test_test_metric(self, mocker, model, fxt_seg_data_entity) -> None:
        model.eval()
        model.on_validation_start()
        mocker_update_loss = mocker.patch.object(
            model,
            "_convert_pred_entity_to_compute_metric",
            return_value=[
                {"preds": torch.randint(0, 2, size=[1, 3, 3]), "target": torch.randint(0, 2, size=[1, 3, 3])},
            ],
        )
        model.test_step(fxt_seg_data_entity[2], 0)
        mocker_update_loss.assert_called_once()

    def test_convert_pred_entity_to_compute_metric(self, model, fxt_seg_data_entity) -> None:
        pred_entity = fxt_seg_data_entity[2]
        out = model._convert_pred_entity_to_compute_metric(pred_entity, fxt_seg_data_entity[2])
        assert isinstance(out, list)
        assert "preds" in out[-1]
        assert "target" in out[-1]
        assert out[-1]["preds"].sum() == out[-1]["target"].sum()
