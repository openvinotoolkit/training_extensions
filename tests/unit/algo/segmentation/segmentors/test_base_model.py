# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
from otx.algo.segmentation.segmentors.base_model import BaseSegmModel
from otx.core.data.entity.base import ImageInfo


class TestBaseSegmModel:
    @pytest.fixture()
    def model(self):
        backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=3, padding=1))
        decode_head = torch.nn.Sequential(torch.nn.Conv2d(64, 2, kernel_size=1))
        decode_head.num_classes = 3
        return BaseSegmModel(backbone, decode_head)

    @pytest.fixture()
    def inputs(self):
        inputs = torch.randn(1, 3, 256, 256)
        masks = torch.randint(0, 2, (1, 256, 256))
        return inputs, masks

    def test_forward_returns_tensor(self, model, inputs):
        images = inputs[0]
        output = model.forward(images)
        assert isinstance(output, torch.Tensor)

    def test_forward_returns_loss(self, model, inputs):
        model.criterion.name = "CrossEntropyLoss"
        images, masks = inputs
        img_metas = [ImageInfo(img_shape=(256, 256), img_idx=0, ori_shape=(256, 256))]
        output = model.forward(images, img_metas=img_metas, masks=masks, mode="loss")
        assert isinstance(output, dict)
        assert "CrossEntropyLoss" in output

    def test_forward_returns_prediction(self, model, inputs):
        images = inputs[0]
        output = model.forward(images, mode="predict")
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 256, 256)

    def test_extract_features(self, model, inputs):
        images = inputs[0]
        features = model.extract_features(images)
        assert isinstance(features, tuple)
        assert isinstance(features[0], torch.Tensor)
        assert isinstance(features[1], torch.Tensor)
        assert features[1].shape == (1, 2, 256, 256)

    def test_calculate_loss(self, model, inputs):
        model.criterion.name = "CrossEntropyLoss"
        images, masks = inputs
        img_metas = [ImageInfo(img_shape=(256, 256), img_idx=0, ori_shape=(256, 256))]
        loss = model.calculate_loss(images, img_metas, masks, interpolate=False)
        assert isinstance(loss, dict)
        assert "CrossEntropyLoss" in loss
        assert isinstance(loss["CrossEntropyLoss"], torch.Tensor)

    def test_get_valid_label_mask(self, model):
        img_metas = [ImageInfo(img_shape=(256, 256), img_idx=0, ignored_labels=[0, 2], ori_shape=(256, 256))]
        valid_label_mask = model.get_valid_label_mask(img_metas)
        assert isinstance(valid_label_mask, list)
        assert len(valid_label_mask) == 1
        assert isinstance(valid_label_mask[0], torch.Tensor)
        assert valid_label_mask[0].tolist() == [0, 1, 0]
