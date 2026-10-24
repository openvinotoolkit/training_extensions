# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DEIMV2 detection model."""

from __future__ import annotations

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.deimv2 import DEIMV2
from getitune.data.entity.sample import PredictionBatch


class TestDEIMV2:
    """Test class for DEIMV2 detection model."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "deimv2_s",
            "deimv2_m",
        ],
    )
    def test_init(self, model_name: str) -> None:
        """Test DEIMV2 model initialization."""
        model = DEIMV2(
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
        """Test DEIMV2 model creation."""
        model = DEIMV2(
            model_name="deimv2_s",
            label_info=10,
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
        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        created_model = model._create_model()

        # Check optimizer configuration exists
        assert hasattr(created_model, "optimizer_configuration")
        assert len(created_model.optimizer_configuration) == 3

    @pytest.mark.parametrize(
        ("model_name", "expected_lr"),
        [
            ("deimv2_x", 0.00001),
            ("deimv2_s", 0.000025),
        ],
    )
    def test_backbone_lr_values(self, model_name: str, expected_lr: float) -> None:
        """Test that backbone learning rates are correctly set for each model variant."""
        model = DEIMV2(
            model_name=model_name,
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        created_model = model._create_model()

        # Check that the first optimizer config has the expected backbone lr
        assert created_model.optimizer_configuration[0]["lr"] == expected_lr

    def test_loss_computation(self, fxt_detection_batch) -> None:
        """Test DEIMV2 loss computation in training mode."""
        model = DEIMV2(
            model_name="deimv2_s",
            label_info=10,
            pretrained=False,
        )

        # Set model to training mode
        model.train()

        # Forward pass should return loss dictionary
        output = model(fxt_detection_batch)

        # Check that output contains expected DEIM loss components
        assert isinstance(output, dict)
        expected_losses = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_mal"]

        for loss_name in expected_losses:
            assert loss_name in output
            assert isinstance(output[loss_name], torch.Tensor)

    @pytest.mark.parametrize(
        "model_name",
        [
            "deimv2_s",
        ],
    )
    def test_predict(self, model_name: str, fxt_detection_batch) -> None:
        """Test DEIMV2 prediction in evaluation mode."""
        model = DEIMV2(
            model_name=model_name,
            label_info=3,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        # Set model to evaluation mode
        model.eval()

        # Forward pass should return predictions
        output = model(fxt_detection_batch)

        # Check that output is PredictionBatch
        assert isinstance(output, PredictionBatch)
        assert output.batch_size == 2

    @pytest.mark.parametrize(
        "model_name",
        [
            "deimv2_s",
        ],
    )
    def test_export(self, model_name: str) -> None:
        """Test DEIMV2 export functionality."""
        model = DEIMV2(
            model_name=model_name,
            label_info=3,
            pretrained=False,
        )

        # Set model to evaluation mode
        model.eval()

        # Test export forward pass
        output = model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 3  # Should return boxes, scores, labels

        # Test with explain mode
        model.explain_mode = True
        output = model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 5  # Should return boxes, scores, labels, saliency_map, feature_vector

    def test_dinov3_backbone(self) -> None:
        """Test that DEIMV2 uses DINOv3STA backbone."""
        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        created_model = model._create_model()

        # Check that backbone is DINOv3STAsModule
        from getitune.backend.lightning.models.detection.backbones.dinov3sta import DINOv3STAsModule

        assert isinstance(created_model.backbone, DINOv3STAsModule)

    def test_hybrid_encoder(self) -> None:
        """Test that DEIMV2 uses HybridEncoder."""
        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        created_model = model._create_model()

        # Check that encoder is HybridEncoderModule
        from getitune.backend.lightning.models.detection.necks.dfine_hybrid_encoder import HybridEncoderModule

        assert isinstance(created_model.encoder, HybridEncoderModule)

    def test_deim_transformer_decoder(self) -> None:
        """Test that DEIMV2 uses DEIMTransformer decoder."""
        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        created_model = model._create_model()

        # Check that decoder is DEIMTransformerModule
        from getitune.backend.lightning.models.detection.heads.deim_decoder import DEIMTransformerModule

        assert isinstance(created_model.decoder, DEIMTransformerModule)

    def test_optimizer_configuration_structure(self) -> None:
        """Test optimizer configuration has proper structure."""
        model = DEIMV2(
            model_name="deimv2_s",
            label_info=5,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        created_model = model._create_model()
        opt_config = created_model.optimizer_configuration

        # Should have 3 configurations
        assert len(opt_config) == 3

        # First config: dinov3 params excluding norm/bn/bias
        assert "params" in opt_config[0]
        assert "lr" in opt_config[0]
        assert "dinov3" in opt_config[0]["params"]

        # Second config: dinov3 norm/bn/bias with weight_decay=0
        assert "params" in opt_config[1]
        assert "lr" in opt_config[1]
        assert opt_config[1].get("weight_decay") == 0.0

        # Third config: sta/encoder/decoder norm/bn/bias with weight_decay=0
        assert "params" in opt_config[2]
        assert opt_config[2].get("weight_decay") == 0.0
