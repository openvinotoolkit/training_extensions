import pytest
import torch

from otx.mpa.modules.models.classifiers.task_incremental_classifier import (
    ImageClassifier,
    TaskIncrementalLwF,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestTaskIncrementalLwF:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(ImageClassifier, "__init__", return_value=None)
        self.kwargs = dict(soft_label=torch.rand(1, 1))
        self.incremental_classifier = TaskIncrementalLwF(backbone=None, **self.kwargs)

    @e2e_pytest_unit
    def test_forward_train(self, mocker):
        img = torch.rand(1, 3, 224, 224)
        gt_label = torch.rand(1, 1)
        self.incremental_classifier.extract_feat = mocker.MagicMock()
        self.incremental_classifier.head = mocker.MagicMock()
        losses = self.incremental_classifier.forward_train(img, gt_label, **self.kwargs)

        assert losses is not None
