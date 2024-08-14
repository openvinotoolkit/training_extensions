# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore
from otx.algo.segmentation.segmentors import BaseSegmModel, MeanTeacher
from otx.core.data.entity.base import ImageInfo
from torch import nn


class TestMeanTeacher:
    @pytest.fixture()
    def model(self):
        decode_head = nn.Conv2d(3, 2, 1)
        decode_head.num_classes = 2
        loss = CrossEntropyLossWithIgnore(ignore_index=255)
        model = BaseSegmModel(
            backbone=nn.Sequential(nn.Conv2d(3, 5, 1), nn.ReLU(), nn.Conv2d(5, 3, 1)),
            decode_head=decode_head,
            criterion=loss,
        )
        return MeanTeacher(model)

    @pytest.fixture()
    def inputs(self):
        return torch.randn(4, 3, 10, 10)

    @pytest.fixture()
    def unlabeled_weak_images(self):
        return torch.randn(4, 3, 10, 10)

    @pytest.fixture()
    def unlabeled_strong_images(self):
        return torch.randn(4, 3, 10, 10)

    @pytest.fixture()
    def img_metas(self):
        return [ImageInfo(img_idx=i, img_shape=(10, 10), ori_shape=(10, 10)) for i in range(4)]

    @pytest.fixture()
    def masks(self):
        return torch.randint(0, 2, size=(4, 10, 10)).long()

    def test_forward_labeled_images(self, model, inputs, img_metas):
        output = model.forward(inputs, img_metas, mode="tensor")
        assert output.shape == (4, 2, 10, 10)

    def test_forward_unlabeled_images(
        self,
        model,
        inputs,
        unlabeled_weak_images,
        unlabeled_strong_images,
        img_metas,
        masks,
    ):
        output = model.forward(
            inputs,
            unlabeled_weak_images,
            unlabeled_strong_images,
            img_metas=img_metas,
            unlabeled_img_metas=img_metas,
            global_step=1,
            steps_per_epoch=1,
            masks=masks,
            mode="loss",
        )
        assert isinstance(output, dict)
        assert "loss_ce_ignore" in output
        assert "loss_ce_ignore_unlabeled" in output
        assert isinstance(output["loss_ce_ignore"], torch.Tensor)
        assert isinstance(output["loss_ce_ignore_unlabeled"], torch.Tensor)
        assert output["loss_ce_ignore"] > 0
        assert output["loss_ce_ignore_unlabeled"] > 0

    def test_generate_pseudo_labels(self, model, unlabeled_weak_images):
        pl_from_teacher, reweight_unsup = model._generate_pseudo_labels(
            unlabeled_weak_images,
            percent_unreliable=20,
        )

        assert isinstance(pl_from_teacher, torch.Tensor)
        assert pl_from_teacher.shape == (4, 1, 10, 10)
        assert isinstance(reweight_unsup, torch.Tensor)
        assert isinstance(reweight_unsup.item(), float)
