import pytest
import torch
import torch.nn as nn

from otx.algorithms.classification.adapters.mmcls import ConstrastiveHead
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestConstrastiveHead:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch) -> None:
        class MockNeck(nn.Sequential):
            def __init__(self):
                super().__init__(nn.Linear(2, 2, bias=False))

            def init_weights(self, init_linear=None):
                for module in self.modules():
                    if hasattr(module, "weight") and module.weight is not None:
                        nn.init.constant_(module.weight, 1)

        def build_mock_neck(*args, **kwargs):
            return MockNeck()

        monkeypatch.setattr(
            "otx.algorithms.classification.adapters.mmcls.models.heads.contrastive_head.build_neck", build_mock_neck
        )

        self.inputs = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.targets = torch.Tensor([[2.0, 3.0], [4.0, 5.0]])

    @e2e_pytest_unit
    def test_forward(self) -> None:
        """Test forward function."""
        contrastive_head = ConstrastiveHead(predictor={})
        contrastive_head.init_weights()

        result = contrastive_head(self.inputs, self.targets)
        expected_result = {"loss": torch.tensor(0.0255)}

        assert torch.allclose(result["loss"], expected_result["loss"], rtol=1e-4, atol=1e-4)

    @e2e_pytest_unit
    def test_forward_no_size_average(self) -> None:
        """Test forward function without size averaging."""
        contrastive_head = ConstrastiveHead(predictor={}, size_average=False)
        contrastive_head.init_weights()

        result = contrastive_head(self.inputs, self.targets)
        expected_result = {"loss": torch.tensor(0.0511)}

        assert torch.allclose(result["loss"], expected_result["loss"], rtol=1e-4, atol=1e-4)
