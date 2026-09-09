# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for instance segmentation model entity."""

from __future__ import annotations

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.instance_segmentation.base import LightningInstanceSegModel
from getitune.backend.lightning.models.instance_segmentation.maskrcnn_tv import MaskRCNNTV
from getitune.backend.lightning.tools.explain.explain_algo import feature_vector_fn
from getitune.types.export import TaskLevelExportParameters


class TestLightningInstanceSegModel:
    @pytest.fixture
    def model(self) -> LightningInstanceSegModel:
        return MaskRCNNTV(
            label_info=1,
            model_name="maskrcnn_resnet_50",
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    def test_get_explain_fn(self, model):
        model.explain_mode = True
        explain_fn = model.get_explain_fn()
        assert callable(explain_fn)

    def test_forward_explain_inst_seg(self, model, fxt_inst_seg_data_entity):
        inputs = fxt_inst_seg_data_entity[2]
        inputs.images = torch.randn(1, 3, 224, 224)
        model.model.feature_vector_fn = feature_vector_fn
        model.model.explain_fn = model.get_explain_fn()
        model.eval()
        result = model._forward_explain_inst_seg(model.model, inputs, mode="predict")

        assert "predictions" in result
        assert "feature_vector" in result
        assert "saliency_map" in result

    def test_customize_inputs(self, model, fxt_inst_seg_data_entity) -> None:
        output_data = model._customize_inputs(fxt_inst_seg_data_entity[2])
        assert output_data["entity"] == fxt_inst_seg_data_entity[2]

    def test_forward_explain(self, model, fxt_inst_seg_data_entity):
        inputs = fxt_inst_seg_data_entity[2]
        inputs.images = [image.float() for image in inputs.images]
        model.training = False
        model.eval()
        model.explain_mode = True
        outputs = model.forward_explain(inputs)

        assert outputs.saliency_map is not None
        assert len(outputs.saliency_map) > 0
        assert outputs.feature_vector is not None
        assert outputs.saliency_map is not None

    def test_reset_restore_model_forward(self, model):
        model.explain_mode = True
        initial_model_forward = model.model.forward

        model._reset_model_forward()
        assert model.original_model_forward is not None
        assert str(model.model.forward) != str(model.original_model_forward)

        model._restore_model_forward()
        assert model.original_model_forward is None
        assert str(model.model.forward) == str(initial_model_forward)

    def test_export_parameters(self, model):
        parameters = model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.task_type == "instance_segmentation"
        assert parameters.nms_execute is False

    def test_export_parameters_with_nms(self, model):
        model.export_nms = True

        parameters = model._export_parameters

        assert parameters.nms_execute is False

    def test_forward_for_tracing_forwards_explain_mode(self, model, mocker):
        observed_arguments = []

        def export_model(_inputs, _meta_info_list, explain_mode=False, with_nms=False) -> tuple[()]:
            observed_arguments.append((explain_mode, with_nms))
            return ()

        mocker.patch.object(model.model, "export", export_model)
        LightningInstanceSegModel.forward_for_tracing(model, torch.randn(1, 3, 224, 224))

        assert observed_arguments == [(False, True)]

    @pytest.mark.parametrize("export_nms", [False, True])
    def test_forward_for_tracing_always_forwards_nms(self, model, mocker, export_nms):
        observed_with_nms = []

        def export_model(_inputs, _meta_info_list, explain_mode=False, with_nms=False) -> tuple[()]:
            observed_with_nms.append(with_nms)
            return ()

        mocker.patch.object(model.model, "export", export_model)
        model.export_nms = export_nms

        LightningInstanceSegModel.forward_for_tracing(model, torch.randn(1, 3, 224, 224))

        assert observed_with_nms == [True]

    def test_export_nms_defaults_to_false(self, model):
        assert model.export_nms is False

    def test_export_nms_can_be_enabled(self):
        model = MaskRCNNTV(
            label_info=1,
            model_name="maskrcnn_resnet_50",
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
            export_nms=True,
        )
        assert model.export_nms is True

    def test_dummy_input(self, model):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
