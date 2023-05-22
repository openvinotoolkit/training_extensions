import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg.models.losses import (
    PixelPrototypeCELoss,
)

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestPixelPrototypeCELoss:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.loss_proto_ce = PixelPrototypeCELoss()

    @e2e_pytest_unit
    def test_forward(self):
        dummy_out = torch.rand(4, 5, 512, 512)
        proto_logits = torch.rand(1280, 16)
        proto_target = torch.rand(1280)
        target = torch.randint(low=0, high=5, size=(4, 1, 512, 512))
        loss = self.loss_proto_ce(dummy_out, proto_logits, proto_target, target)
        assert loss is not None
        assert loss >= 0
