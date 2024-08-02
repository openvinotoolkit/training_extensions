# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for instance segmentation model entity."""

import pytest
import torch
from otx.algo.explain.explain_algo import feature_vector_fn
from otx.algo.instance_segmentation.maskrcnn import MaskRCNNEfficientNet
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel
from otx.core.types.export import TaskLevelExportParameters


class TestOTXInstanceSegModel:
    @pytest.fixture()
    def otx_model(self) -> ExplainableOTXInstanceSegModel:
        return MaskRCNNEfficientNet(label_info=1)

    def test_create_model(self, otx_model) -> None:
        mmdet_model = otx_model._create_model()
        assert mmdet_model is not None
        assert isinstance(mmdet_model, torch.nn.Module)

    def test_get_explain_fn(self, otx_model):
        otx_model.explain_mode = True
        explain_fn = otx_model.get_explain_fn()
        assert callable(explain_fn)

    def test_forward_explain_inst_seg(self, otx_model, fxt_inst_seg_data_entity):
        inputs = fxt_inst_seg_data_entity[2]
        inputs.images = torch.randn(1, 3, 224, 224)
        otx_model.model.feature_vector_fn = feature_vector_fn
        otx_model.model.explain_fn = otx_model.get_explain_fn()
        result = otx_model._forward_explain_inst_seg(otx_model.model, inputs, mode="predict")

        assert "predictions" in result
        assert "feature_vector" in result
        assert "saliency_map" in result

    def test_customize_inputs(self, otx_model, fxt_inst_seg_data_entity) -> None:
        output_data = otx_model._customize_inputs(fxt_inst_seg_data_entity[2])
        assert output_data["mode"] == "loss"
        assert output_data["entity"] == fxt_inst_seg_data_entity[2]

    def test_forward_explain(self, otx_model, fxt_inst_seg_data_entity):
        inputs = fxt_inst_seg_data_entity[2]
        inputs.images = [image.float() for image in inputs.images]
        otx_model.training = False
        otx_model.explain_mode = True
        outputs = otx_model.forward_explain(inputs)

        assert outputs.has_xai_outputs
        assert outputs.feature_vector is not None
        assert outputs.saliency_map is not None

    def test_reset_restore_model_forward(self, otx_model):
        otx_model.explain_mode = True
        initial_model_forward = otx_model.model.forward

        otx_model._reset_model_forward()
        assert otx_model.original_model_forward is not None
        assert str(otx_model.model.forward) != str(otx_model.original_model_forward)

        otx_model._restore_model_forward()
        assert otx_model.original_model_forward is None
        assert str(otx_model.model.forward) == str(initial_model_forward)

    def test_export_parameters(self, otx_model):
        parameters = otx_model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.task_type == "instance_segmentation"

    def test_dummy_input(self, otx_model):
        batch_size = 2
        batch = otx_model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
