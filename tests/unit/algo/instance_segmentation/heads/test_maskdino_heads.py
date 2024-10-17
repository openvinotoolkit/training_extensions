# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test for various MaskDINO heads."""
import pytest
import torch
from otx.algo.instance_segmentation.maskdino import MaskDINO
from otx.algo.instance_segmentation.heads import MaskDINOEncoderHead, MaskDINODecoderHead
from otx.algo.instance_segmentation.losses import MaskDINOCriterion


class TestMaskDINOTransformerHeads:
    @pytest.fixture
    def fxt_shape_spec(self):
        model = MaskDINO(label_info=3, model_name="resnet50")
        _, specs = model._build_backbone()
        return specs

    def test_maskdino_encoder_decoder_head(self, fxt_shape_spec, num_classes=2):
        maskdino_encoder_head = MaskDINOEncoderHead("resnet50", fxt_shape_spec)
        maskdino_decoder_head = MaskDINODecoderHead("resnet50", num_classes=num_classes)
        criterion = MaskDINOCriterion(num_classes=2)
        features = {
            "res2": torch.randn(1, 256, 256, 256),
            "res3": torch.randn(1, 512, 128, 128),
            "res4": torch.randn(1, 1024, 64, 64),
            "res5": torch.randn(1, 2048, 32, 32),
        }

        targets = [
            {
                "boxes": torch.randn(10, 4),
                "labels": torch.randint(0, num_classes, (10,)),
                "masks": torch.ones(10, 512, 512),
            }
        ]

        mask_features, _, multi_scale_features = maskdino_encoder_head(features)
        outputs, mask_dict = maskdino_decoder_head(multi_scale_features, mask_features, targets=targets)
        assert isinstance(outputs, dict)
        assert "pred_masks" in outputs
        assert "pred_logits" in outputs
        assert "pred_boxes" in outputs
        assert "aux_outputs" in outputs
        assert "interm_outputs" in outputs
        assert isinstance(mask_dict, dict)
        assert "known_indice" in mask_dict
        assert "map_known_indice" in mask_dict
        assert "known_lbs_bboxes" in mask_dict
        assert "output_known_lbs_bboxes" in mask_dict
        assert "pad_size" in mask_dict

        losses = criterion(outputs, targets, mask_dict)
        assert isinstance(losses, dict)
        assert "loss_mask" in losses
        assert "loss_ce" in losses
        assert "loss_giou" in losses
        assert "loss_bbox" in losses
        assert "loss_dice" in losses