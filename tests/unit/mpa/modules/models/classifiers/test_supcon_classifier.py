import numpy as np
import pytest
import torch
from copy import deepcopy

from otx.mpa.cls.inferrer import ClsInferrer
from otx.mpa.modules.models.classifiers.supcon_classifier import SupConClassifier, ImageClassifier
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestSupConClassifier:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(ImageClassifier, "__init__", return_value=None)
        self.kwargs = dict(multilabel=False, hierarchical=False)
        self.supcon_classifier = SupConClassifier(backbone=None, **self.kwargs)

    @e2e_pytest_unit
    def test_forward_train(self, mocker):
        img = torch.rand(1, 3, 224, 224)
        gt_label = torch.rand(1, 1)
        self.supcon_classifier.extract_feat = mocker.MagicMock()
        self.supcon_classifier.head = mocker.MagicMock()
        losses = self.supcon_classifier.forward_train(img, gt_label, **self.kwargs)

        assert losses is not None