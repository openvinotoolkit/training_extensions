# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for instance segmentation model entity."""

import pytest
import torch
from otx.algo.explain.explain_algo import feature_vector_fn
from otx.algo.instance_segmentation.maskrcnn import MaskRCNN
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntityWithXAI
from otx.core.model.entity.instance_segmentation import MMDetInstanceSegCompatibleModel


class TestOTXInstanceSegModel:
    @pytest.fixture()
    def otx_model(self) -> MMDetInstanceSegCompatibleModel:
        return MaskRCNN(num_classes=1, variant="efficientnetb2b")

    def test_create_model(self, otx_model) -> None:
        mmdet_model = otx_model._create_model()
        assert mmdet_model is not None
        assert isinstance(mmdet_model, torch.nn.Module)

    def test_get_explain_fn(self, otx_model):
        otx_model.explain_mode = True
        explain_fn = otx_model.get_explain_fn()
        assert callable(explain_fn)

    def test_forward_explain_inst_seg(self, otx_model, fxt_data_sample):
        inputs = torch.randn(1, 3, 224, 224)
        otx_model.model.feature_vector_fn = feature_vector_fn
        otx_model.model.explain_fn = otx_model.get_explain_fn()
        result = otx_model._forward_explain_inst_seg(otx_model.model, inputs, fxt_data_sample, mode="predict")

        assert "predictions" in result
        assert "feature_vector" in result
        assert "saliency_map" in result

    def test_customize_inputs(self, otx_model, fxt_inst_seg_data_entity) -> None:
        output_data = otx_model._customize_inputs(fxt_inst_seg_data_entity[2])
        assert output_data is not None
        assert "gt_instances" in output_data["data_samples"][-1]
        assert "masks" in output_data["data_samples"][-1].gt_instances
        assert output_data["data_samples"][-1].metainfo["pad_shape"] == output_data["inputs"].shape[-2:]

    def test_forward_explain(self, otx_model, fxt_inst_seg_data_entity):
        inputs = fxt_inst_seg_data_entity[2]
        otx_model.training = False
        otx_model.explain_mode = True
        outputs = otx_model.forward_explain(inputs)

        assert isinstance(outputs, InstanceSegBatchPredEntityWithXAI)
        assert outputs.feature_vectors is not None
        assert outputs.saliency_maps is not None

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
        otx_model.image_size = (1, 64, 64, 3)
        otx_model.explain_mode = False
        parameters = otx_model._export_parameters
        assert isinstance(parameters, dict)
        assert "output_names" in parameters

        otx_model.explain_mode = True
        parameters = otx_model._export_parameters
        assert parameters["output_names"] == ["feature_vector", "saliency_map"]
