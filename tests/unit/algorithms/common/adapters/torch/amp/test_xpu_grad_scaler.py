# test_xpu_grad_scaler.py

import pytest
import torch
from otx.algorithms.common.utils import is_xpu_available
if is_xpu_available():
    from otx.algorithms.common.adapters.torch.amp.xpu_grad_scaler import XPUGradScaler


@pytest.mark.skipif(not is_xpu_available(), reason="XPU is not available")
class TestXPUGradScaler:
    @pytest.fixture
    def grad_scaler(self):
        return XPUGradScaler()

    @pytest.fixture
    def optimizer(self):
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return optimizer

    def test_init(self, grad_scaler):
        assert grad_scaler._enabled
        assert grad_scaler._init_scale == 2.0**16
        assert grad_scaler._growth_factor == 2.0
        assert grad_scaler._backoff_factor == 0.5
        assert grad_scaler._growth_interval == 2000

    def test_scale(self, grad_scaler):
        outputs = torch.tensor([1.0, 2.0, 3.0], device="xpu:0")
        scaled_outputs = grad_scaler.scale(outputs)
        assert scaled_outputs.device.type == "xpu"
        assert torch.equal(scaled_outputs, outputs * grad_scaler._scale)

    def test_unscale_grads(self, grad_scaler, optimizer):
        inv_scale = 1.0
        found_inf = False
        output = grad_scaler._unscale_grads_(optimizer, inv_scale, found_inf, allow_bf16=False)
        assert isinstance(output, dict)
        assert not output
