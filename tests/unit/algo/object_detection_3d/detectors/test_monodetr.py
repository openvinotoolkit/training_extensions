# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test MonoDetr."""

import pytest
import torch
from otx.algo.object_detection_3d.backbones.monodetr_resnet import BackboneBuilder
from otx.algo.object_detection_3d.detectors.monodetr import MonoDETR
from otx.algo.object_detection_3d.heads.depth_predictor import DepthPredictor
from otx.algo.object_detection_3d.heads.depthaware_transformer import DepthAwareTransformerBuilder


class TestMonoDETR:
    @pytest.fixture()
    def model(self):
        backbone = BackboneBuilder("monodetr_50")
        # transformer
        depthaware_transformer = DepthAwareTransformerBuilder("monodetr_50")
        # depth prediction module
        depth_predictor = DepthPredictor(depth_num_bins=80, depth_min=1e-3, depth_max=60.0, hidden_dim=256)

        num_classes = 2
        num_queries = 50
        num_feature_levels = 4
        return MonoDETR(
            backbone,
            depthaware_transformer,
            depth_predictor,
            num_classes=num_classes,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            with_box_refine=True,
        )

    def test_monodetr_forward(self, model):
        # Create a sample input
        images = torch.randn(2, 3, 224, 224)
        calibs = torch.randn(2, 3, 4)
        img_sizes = torch.tensor([[224, 224], [224, 224]])
        # Perform forward pass
        output = model(images, calibs, img_sizes, mode="predict")

        # Check the output
        assert "scores" in output
        assert "boxes_3d" in output
        assert "size_3d" in output
        assert "depth" in output
        assert "heading_angle" in output
        assert "pred_depth_map_logits" in output
        assert "aux_outputs" in output

        # Check the shape of the output tensors
        assert output["scores"].shape == (2, 550, 2)
        assert output["boxes_3d"].shape == (2, 550, 6)
        assert output["size_3d"].shape == (2, 550, 3)
        assert output["depth"].shape == (2, 550, 2)
        assert output["heading_angle"].shape == (2, 550, 24)
        assert output["pred_depth_map_logits"].shape == (2, 81, 14, 14)

        # Check error handling when loss is None
        with pytest.raises(ValueError):  # noqa: PT011
            output = model(images, calibs, img_sizes, mode="loss")

        # Check the export mode
        export_output = model(images, calibs, img_sizes, mode="export")
        assert "scores" in export_output
        assert "boxes_3d" in export_output
        assert export_output["scores"].shape == (2, 550, 2)
        assert export_output["scores"].min() >= 0
        assert export_output["scores"].max() <= 1
