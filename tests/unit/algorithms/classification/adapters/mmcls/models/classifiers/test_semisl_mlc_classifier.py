# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.classifiers.semisl_multilabel_classifier import (
    SAMImageClassifier,
    SemiSLMultilabelClassifier,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSemiSLMultilabelClassifier:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(SAMImageClassifier, "__init__", return_value=None)
        self.semisl_classifier = SemiSLMultilabelClassifier()
        self.kwargs = dict()

    @e2e_pytest_unit
    def test_forward_train(self, mocker):
        img = torch.rand(1, 3, 224, 224)
        self.kwargs["gt_label"] = torch.rand(1, 1)
        self.kwargs["extra_0"] = {"img": torch.rand(1, 3, 224, 224), "img_strong": torch.rand(1, 3, 224, 224)}
        self.kwargs["img_strong"] = torch.rand(1, 3, 224, 224)
        self.semisl_classifier.extract_feat = mocker.MagicMock()
        self.semisl_classifier.head = mocker.MagicMock()
        losses = self.semisl_classifier.forward_train(img, **self.kwargs)

        assert losses is not None
