# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MaskDINO architecture."""

import pytest
import torch
from otx.algo.instance_segmentation.maskdino import MaskDINO
from otx.algo.instance_segmentation.segmentors.maskdino import MaskDINOHead
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity


class TestMaskDINO:
    @pytest.mark.parametrize("model", [MaskDINO(label_info=3, model_name="resnet50")])
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 320, 320])
        data.masks = [torch.ones((len(masks), 320, 320)) for masks in data.masks]
        model(data)

    @pytest.mark.parametrize("model", [MaskDINO(label_info=3, model_name="resnet50")])
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn(2, 3, 320, 320)
        model.eval()
        output = model(data)
        assert isinstance(output, InstanceSegBatchPredEntity)

    @pytest.mark.parametrize("model", [MaskDINO(label_info=3, model_name="resnet50")])
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 320, 320))
        assert len(output) == 3

    def test_roi_mask_extraction(self):
        maskdino_head = MaskDINOHead(num_classes=1, pixel_decoder=None, predictor=None)
        extracted_roi = maskdino_head.roi_mask_extraction(torch.randn(10, 4), torch.randn(10, 320, 320))
        assert extracted_roi.shape == (10, 28, 28)

    def test_post_process(self, mocker):
        outputs = {
            "pred_logits": torch.randn(1, 100, 1),
            "pred_masks": torch.randn(1, 100, 256, 256),
            "pred_boxes": torch.randn(1, 100, 4),
        }

        mocker.patch(
            "otx.algo.instance_segmentation.segmentors.maskdino.MaskDINOHead.forward",
            return_value=[outputs, None],
        )

        maskdino_head = MaskDINOHead(num_classes=1, pixel_decoder=None, predictor=None)

        image_info = ImageInfo(
            img_idx=0,
            img_shape=(320, 320),
            ori_shape=(320, 320),
            ignored_labels=[],
        )

        batch_bboxes_scores, batch_labels, batch_masks = maskdino_head.predict(None, [image_info])

        assert len(batch_bboxes_scores) == len(batch_labels) == len(batch_masks)
        assert len(batch_bboxes_scores[0]) == len(batch_labels[0]) == len(batch_masks[0])
        assert batch_masks[0].shape[-2:] == (320, 320)
        assert batch_bboxes_scores[0].shape[-1:] == (5,)
