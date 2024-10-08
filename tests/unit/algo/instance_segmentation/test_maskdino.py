# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MaskDINO architecture."""

import pytest
import torch
from otx.algo.instance_segmentation.maskdino import MaskDINO, MaskDINOR50
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity


class TestMaskDINO:
    @pytest.mark.parametrize("model", [MaskDINOR50(3, "maskdino_resnet_50")])
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 320, 320])
        data.masks = [torch.ones((len(masks), 320, 320)) for masks in data.masks]
        model(data)

    @pytest.mark.parametrize("model", [MaskDINOR50(3, "maskdino_resnet_50")])
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 320, 320), torch.randn(3, 320, 320)]
        model.eval()
        output = model(data)
        assert isinstance(output, InstanceSegBatchPredEntity)

    @pytest.mark.parametrize("model", [MaskDINOR50(3, "maskdino_resnet_50")])
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 320, 320))
        assert len(output) == 3

    def test_roi_mask_extraction(self):
        model = MaskDINO.from_config(num_classes=1)
        extracted_roi = model.roi_mask_extraction(torch.randn(10, 4), torch.randn(10, 320, 320))
        assert extracted_roi.shape == (1, 10, 28, 28)

    def test_post_process(self):
        model = MaskDINOR50(1, "maskdino_resnet_50")
        outputs = {
            "pred_logits": torch.randn(1, 100, 1),
            "pred_masks": torch.randn(1, 100, 256, 256),
            "pred_boxes": torch.randn(1, 100, 4),
        }

        image_info = ImageInfo(
            img_idx=0,
            img_shape=(320, 320),
            ori_shape=(320, 320),
            ignored_labels=[],
        )

        masks, bboxes, labels, scores = model.post_process_instance_segmentation(
            outputs,
            [image_info],
        )

        assert len(masks) == len(bboxes) == len(labels) == len(scores) == 1
        assert len(bboxes[0]) == len(masks[0]) == len(labels[0]) == len(scores[0])
        assert masks[0].shape[-2:] == (320, 320)
        assert bboxes[0].shape[-1:] == (4, )
