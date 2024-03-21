from unittest.mock import patch

import pytest
import torch
from otx.algo.detection.losses.cross_focal_loss import CrossSigmoidFocalLoss, OrdinaryFocalLoss


class TestCrossFocalLoss:
    @pytest.fixture()
    def mock_tensor(self) -> torch.Tensor:
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    @pytest.fixture()
    def mock_targets(self) -> torch.Tensor:
        return torch.tensor([0, 1], dtype=torch.long)

    @pytest.fixture()
    def mock_weights(self) -> torch.Tensor:
        return torch.tensor([0.5, 0.5], dtype=torch.float32)

    @pytest.fixture()
    def mock_valid_label_mask(self) -> torch.Tensor:
        return torch.tensor([1, 1], dtype=torch.float32)

    def test_cross_focal_forward(self, mock_tensor, mock_targets, mock_weights, mock_valid_label_mask):
        with patch(
            "otx.algo.detection.losses.cross_focal_loss.py_sigmoid_focal_loss",
            return_value=torch.tensor(0.5),
        ) as mock_loss_func:
            loss_fn = CrossSigmoidFocalLoss()

            loss = loss_fn(
                pred=mock_tensor,
                targets=mock_targets,
                weight=mock_weights,
                valid_label_mask=mock_valid_label_mask,
                reduction_override="mean",
            )
            mock_loss_func.assert_called()
            assert loss.item() == pytest.approx(0.5, 0.1), "Loss did not match expected value."

    def test_ordinary_focal_forward(self, mock_tensor, mock_targets, mock_weights):
        loss_fn = OrdinaryFocalLoss(gamma=1.5)

        loss = loss_fn(inputs=mock_tensor, targets=mock_targets, label_weights=mock_weights)

        assert loss >= 0, "Loss should be non-negative."
