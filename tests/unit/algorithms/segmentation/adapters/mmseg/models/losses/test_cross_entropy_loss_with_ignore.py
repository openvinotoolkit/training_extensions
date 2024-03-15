import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg.models.losses.cross_entropy_loss_with_ignore import (
    CrossEntropyLossWithIgnore,
)
from mmseg.models.losses import CrossEntropyLoss

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCrossEntropyLosWithIgnore:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_score = torch.rand([1, 2, 5, 5])
        self.mock_gt = torch.zeros((1, 5, 5), dtype=torch.long)
        self.mock_gt[::2, 1::2, :] = 1
        self.mock_gt[1::2, ::2, :] = 1

        self.loss_f = CrossEntropyLossWithIgnore()

    @e2e_pytest_unit
    def test_is_label_ignored(self):
        loss = self.loss_f(self.mock_score, self.mock_gt, reduction_override="none")
        assert type(loss) == torch.Tensor

        mock_valid_label_mask = torch.Tensor([[1, 0]])
        loss_ignore = self.loss_f(
            self.mock_score, self.mock_gt, reduction_override="none", valid_label_mask=mock_valid_label_mask
        )

        assert torch.all(loss_ignore[::2, 1::2, :] == 0)
        assert torch.all(loss_ignore[1::2, ::2, :] == 0)

        assert torch.equal(loss, loss_ignore) is False

    @e2e_pytest_unit
    def test_is_equal_to_ce_loss(self):
        loss_f_mmseg = CrossEntropyLoss()

        loss_1 = loss_f_mmseg(self.mock_score, self.mock_gt)
        loss_2 = self.loss_f(self.mock_score, self.mock_gt)
        loss_3 = self.loss_f(self.mock_score, self.mock_gt, valid_label_mask=torch.Tensor([1, 1]))

        assert loss_1 == loss_2
        assert loss_2 == loss_3
