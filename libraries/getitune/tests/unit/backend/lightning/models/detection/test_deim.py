# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DEIM DFine detection model."""

from __future__ import annotations

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.deim import DEIMDFine
from getitune.data.entity.sample import PredictionBatch


class TestDEIMDFine:
    """Test class for DEIM DFine detection model."""

    @pytest.fixture(params=["deim_dfine_hgnetv2_n"])
    def fxt_model(self, request) -> DEIMDFine:
        return DEIMDFine(
            model_name=request.param,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "deim_dfine_hgnetv2_n",
            "deim_dfine_hgnetv2_s",
        ],
    )
    def test_init(self, model_name: str) -> None:
        """Test DEIM DFine model initialization."""
        model = DEIMDFine(
            model_name=model_name,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        assert model.model_name == model_name
        assert model.num_classes == 3
        assert model.data_input_params.input_size == (640, 640)
        assert model.input_size_multiplier == 32
        assert model_name in model.pretrained_urls

    def test_create_model(self) -> None:
        """Test DEIM DFine model creation."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=10,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        created_model = model._create_model()
        assert created_model is not None
        assert isinstance(created_model, torch.nn.Module)

        # Check if the model has the expected components
        assert hasattr(created_model, "backbone")
        assert hasattr(created_model, "encoder")
        assert hasattr(created_model, "decoder")
        assert hasattr(created_model, "criterion")
        assert hasattr(created_model, "num_classes")
        assert created_model.num_classes == 10

    def test_backbone_lr_mapping(self) -> None:
        """Test that backbone learning rate mapping works correctly."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_n",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        created_model = model._create_model()

        # Check optimizer configuration exists
        assert hasattr(created_model, "optimizer_configuration")
        assert len(created_model.optimizer_configuration) > 0

        # For 'n' variant, should have 3 configurations
        if model.model_name == "deim_dfine_hgnetv2_n":
            assert len(created_model.optimizer_configuration) == 3
        else:
            assert len(created_model.optimizer_configuration) == 2

    def test_loss_computation(self, fxt_model, fxt_detection_batch) -> None:
        """Test DEIM DFine loss computation in training mode."""
        fxt_model.train()

        # Forward pass should return loss dictionary
        output = fxt_model(fxt_detection_batch)

        # Check that output contains expected DEIM loss components
        assert isinstance(output, dict)
        expected_losses = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_mal"]

        for loss_name in expected_losses:
            assert loss_name in output
            assert isinstance(output[loss_name], torch.Tensor)

    def test_predict(self, fxt_model, fxt_detection_batch) -> None:
        """Test DEIM DFine prediction in evaluation mode."""
        fxt_model.eval()

        # Forward pass should return predictions
        output = fxt_model(fxt_detection_batch)

        # Check that output is PredictionBatch
        assert isinstance(output, PredictionBatch)
        assert output.batch_size == 2

    def test_export(self, fxt_model) -> None:
        """Test DEIM DFine export functionality."""
        fxt_model.eval()

        # Test export forward pass
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 3  # Should return boxes, scores, labels

        # Test with explain mode
        fxt_model.explain_mode = True
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 5  # Should return boxes, scores, labels, saliency_map, feature_vector

    def test_multi_scale_training(self) -> None:
        """Test DEIM DFine with multi-scale training enabled."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            multi_scale=True,
            pretrained=False,
        )

        # Multi-scale should be created in the model
        created_model = model._create_model()
        assert isinstance(created_model.multi_scale, list)
        assert len(created_model.multi_scale) > 0

    def test_weight_dict_configuration(self) -> None:
        """Test that the weight dictionary is properly configured."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        created_model = model._create_model()
        criterion = created_model.criterion

        # Check that weight dict contains expected keys
        expected_weights = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf", "loss_mal"]
        for weight_key in expected_weights:
            assert weight_key in criterion.weight_dict

        # Check specific weight values
        assert criterion.weight_dict["loss_vfl"] == 1
        assert criterion.weight_dict["loss_bbox"] == 5
        assert criterion.weight_dict["loss_giou"] == 2
        assert criterion.weight_dict["loss_fgl"] == 0.15
        assert criterion.weight_dict["loss_ddf"] == 1.5
        assert criterion.weight_dict["loss_mal"] == 1.0

    def test_criterion_parameters(self) -> None:
        """Test that the criterion is configured with correct parameters."""
        model = DEIMDFine(
            model_name="deim_dfine_hgnetv2_s",
            label_info=10,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        created_model = model._create_model()
        criterion = created_model.criterion

        # Check criterion parameters
        assert criterion.alpha == 0.75
        assert criterion.gamma == 1.5
        assert criterion.reg_max == 32
        assert criterion.num_classes == 10
