import pytest
import torch
import pytorch_lightning as pl
from otx.algorithms.anomaly.adapters.anomalib.strategies.xpu_single import SingleXPUStrategy
from otx.algorithms.common.utils.utils import is_xpu_available


@pytest.mark.skipif(not is_xpu_available(), reason="XPU is not available")
class TestSingleXPUStrategy:
    def test_init(self):
        strategy = SingleXPUStrategy(device="xpu:0")
        assert strategy._root_device == "xpu:0"
        assert strategy.accelerator is None

    def test_is_distributed(self):
        strategy = SingleXPUStrategy(device="xpu:0")
        assert not strategy.is_distributed

    def test_setup_optimizers(self):
        strategy = SingleXPUStrategy(device="xpu:0")
        trainer = pl.Trainer()
        # Create mock optimizers and models for testing
        model = torch.nn.Linear(10, 2)
        strategy._optimizers = [torch.optim.Adam(model.parameters(), lr=0.001)]
        strategy._model = model
        trainer.model = model
        strategy.setup_optimizers(trainer)
        assert len(strategy.optimizers) == 1
