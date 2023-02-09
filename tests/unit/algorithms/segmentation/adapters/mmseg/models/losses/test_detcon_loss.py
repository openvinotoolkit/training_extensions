import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg.models.losses.detcon_loss import (
    DetConLoss,
    manual_cross_entropy,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
@pytest.mark.parametrize(
    "logits,labels,weight,expected",
    [
        (
            torch.Tensor([[[-1.2786, 2.1673, -1.0000e09, 6.2263], [-6.7169, -7.1416, 7.3364, -1.0000e09]]]),
            torch.Tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
            torch.Tensor([[1.0, 1.0]]),
            torch.tensor(11.0003),
        ),
        (
            torch.Tensor(
                [[[4.3260e-01, 1.6511e00, -1.0000e09, -7.9047e-01], [9.4888e-01, 2.6476e00, -3.7093e-01, -1.0000e09]]]
            ),
            torch.Tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
            torch.Tensor([[1.0, 1.0]]),
            torch.tensor(0.8755),
        ),
    ],
)
def test_manual_cross_entropy(logits, labels, weight, expected):
    results = manual_cross_entropy(logits, labels, weight)

    assert torch.allclose(results, expected)


@e2e_pytest_unit
@pytest.mark.parametrize(
    "inputs,expected",
    [
        (
            {
                "pred1": torch.Tensor([[[-0.0461, 0.1686, -0.1898, -0.1727], [-0.0609, -0.0948, -0.0463, -0.1461]]]),
                "pred2": torch.Tensor([[[-0.0811, -0.0115, -0.1901, -0.0227], [-0.0230, 0.0898, -0.1495, -0.0360]]]),
                "target1": torch.Tensor([[[-0.7734, 0.1523, 0.4729, -2.2324], [-0.3711, 0.2083, 0.0198, -1.9111]]]),
                "target2": torch.Tensor([[[-1.0625, 1.4160, 0.7988, 1.3291], [-0.2479, 0.9683, 0.0859, 0.4832]]]),
                "pind1": torch.Tensor([[1, 0]]),
                "pind2": torch.Tensor([[1, 0]]),
                "tind1": torch.Tensor([[1, 0]]),
                "tind2": torch.Tensor([[1, 0]]),
            },
            torch.tensor(11.8758),
        )
    ],
)
def test_detcon_loss(mocker, inputs, expected):
    mocker.patch("torch.cuda.device_count", return_value=0)
    detcon_loss = DetConLoss()

    results = detcon_loss(**inputs)

    assert "loss" in results
    assert torch.allclose(results["loss"], expected)
