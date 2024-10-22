# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MaskDINO architecture."""

import pytest
import torch
from otx.algo.instance_segmentation.heads import MaskDINOHead
from otx.algo.instance_segmentation.maskdino import MaskDINO
from otx.algo.utils.mmengine_utils import load_from_http
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

    @pytest.mark.parametrize("model", [MaskDINO(label_info=3, model_name="resnet50")])
    def test_build_shape_specs(self, model):
        out_features = ["res2", "res3", "res4", "res5"]
        strides = [4, 8, 16, 32]
        channels = [256, 512, 1024, 2048]

        shape_specs = model._build_fmap_shape_specs(
            out_features=out_features,
            strides=strides,
            channels=channels,
        )
        assert len(shape_specs) == 4

        for out_feature, stride, channel in zip(out_features, strides, channels):
            assert out_feature in shape_specs
            shape_spec = shape_specs[out_feature]
            assert shape_spec.stride == stride
            assert shape_spec.channels == channel

    @pytest.mark.parametrize("model", [MaskDINO(label_info=3, model_name="resnet50")])
    def test_backbone_weight_loading(self, model):
        pretrained = load_from_http(model.load_from, map_location="cpu")
        tv_backbone = model.model.backbone

        assert torch.allclose(tv_backbone.conv1.weight, pretrained["model"]["backbone.stem.conv1.weight"])
        assert torch.allclose(
            tv_backbone.layer1[0].downsample[0].weight,
            pretrained["model"]["backbone.res2.0.shortcut.weight"],
        )
