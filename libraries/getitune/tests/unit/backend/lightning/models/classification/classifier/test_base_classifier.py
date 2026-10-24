# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from getitune.backend.lightning.models.classification.backbones import EfficientNetBackbone
from getitune.backend.lightning.models.classification.classifier import ImageClassifier
from getitune.backend.lightning.models.classification.heads import LinearClsHead, MultiLabelLinearClsHead
from getitune.backend.lightning.models.classification.losses import AsymmetricAngularLossWithIgnore
from getitune.backend.lightning.models.classification.necks.gap import GlobalAveragePooling


class TestImageClassifier:
    @pytest.fixture(
        params=[
            (LinearClsHead, nn.CrossEntropyLoss, "fxt_multiclass_cls_batch_data_entity"),
            (MultiLabelLinearClsHead, AsymmetricAngularLossWithIgnore, "fxt_multilabel_cls_batch_data_entity"),
        ],
        ids=["multiclass", "multilabel"],
    )
    def fxt_model_and_inputs(self, request):
        head_cls, loss_cls, input_fxt_name = request.param
        backbone = EfficientNetBackbone(model_name="efficientnet_b0")
        neck = GlobalAveragePooling(dim=2)
        head = head_cls(num_classes=3, in_channels=backbone.num_features)
        loss = loss_cls()
        fxt_input = request.getfixturevalue(input_fxt_name)
        fxt_label = (
            torch.stack(fxt_input.labels)
            if isinstance(head, MultiLabelLinearClsHead)
            else torch.cat(fxt_input.labels, dim=0)
        )
        return (
            ImageClassifier(
                backbone=backbone,
                neck=neck,
                head=head,
                loss=loss,
            ),
            fxt_input.images,
            fxt_label,
        )

    def test_forward(self, fxt_model_and_inputs):
        model, images, labels = fxt_model_and_inputs

        output = model(images, labels, mode="tensor")
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 3)

        output = model(images, labels, mode="loss")
        assert isinstance(output, torch.Tensor)

        output = model(images, labels, mode="predict")
        assert isinstance(output, torch.Tensor)

        output = model(images, labels, mode="explain")
        assert isinstance(output, dict)
        assert "logits" in output
        assert "scores" in output
        assert "preds" in output
        assert "saliency_map" in output
        assert "feature_vector" in output

        with pytest.raises(RuntimeError):
            model(images, labels, mode="invalid_mode")
