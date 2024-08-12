# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.algo.classification.backbones import OTXEfficientNet
from otx.algo.classification.classifier import ImageClassifier
from otx.algo.classification.heads import LinearClsHead
from otx.algo.classification.necks.gap import GlobalAveragePooling
from torch import nn


class TestImageClassifier:
    def fxt_classifier_model(self):
        backbone = OTXEfficientNet(version="b0")
        return ImageClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=LinearClsHead(
                num_classes=10,
                in_channels=backbone.num_features,
            ),
            loss=nn.CrossEntropyLoss(reduction="none"),
        )
