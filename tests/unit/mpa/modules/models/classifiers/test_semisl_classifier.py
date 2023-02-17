import numpy as np
import pytest
import torch
from copy import deepcopy

from otx.mpa.cls.inferrer import ClsInferrer
from otx.mpa.modules.models.classifiers.semisl_classifier import SemiSLClassifier, SAMImageClassifier
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestSemiSLClassifier:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(SAMImageClassifier, "__init__", return_value=None)
        self.semisl_classifier = SemiSLClassifier()
        self.kwargs = dict()

    @e2e_pytest_unit
    def test_forward_train(self, mocker):
        img = torch.rand(1, 3, 224, 224)
        self.kwargs["gt_label"] = torch.rand(1, 1)
        self.kwargs["extra_0"] = {"img": torch.rand(1, 3, 224, 224),
                                  "img_strong": torch.rand(1, 3, 224, 224)}
        self.semisl_classifier.extract_feat = mocker.MagicMock()
        self.semisl_classifier.head = mocker.MagicMock()
        losses = self.semisl_classifier.forward_train(img, **self.kwargs)

        assert losses is not None
