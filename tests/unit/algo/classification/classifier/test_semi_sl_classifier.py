# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.classification.backbones import OTXEfficientNet
from otx.algo.classification.classifier import SemiSLClassifier
from otx.algo.classification.heads import SemiSLLinearClsHead
from otx.algo.classification.necks.gap import GlobalAveragePooling


class TestSemiSLClassifier:
    @pytest.fixture()
    def fxt_semi_sl_classifier(self):
        backbone = OTXEfficientNet(version="b0")
        neck = GlobalAveragePooling(dim=2)
        head = SemiSLLinearClsHead(
            num_classes=2,
            in_channels=backbone.num_features,
            use_dynamic_threshold=True,
            min_threshold=0.5,
        )
        loss = torch.nn.CrossEntropyLoss()
        return SemiSLClassifier(backbone, neck, head, loss)

    @pytest.fixture()
    def fxt_inputs(self):
        return {
            "labeled": torch.randn(16, 3, 224, 224),
            "weak_transforms": torch.randn(16, 3, 224, 224),
            "strong_transforms": torch.randn(16, 3, 224, 224),
        }

    def test_extract_feat(self, fxt_semi_sl_classifier, fxt_inputs):
        output = fxt_semi_sl_classifier.extract_feat(fxt_inputs)
        assert isinstance(output, dict)
        assert "labeled" in output
        assert "unlabeled_weak" in output
        assert "unlabeled_strong" in output

    def test_loss(self, fxt_semi_sl_classifier, fxt_inputs):
        labels = torch.randint(0, 2, (16,))
        loss = fxt_semi_sl_classifier.loss(fxt_inputs, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
