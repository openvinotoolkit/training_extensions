"""Lightning strategy for single XPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from otx.algorithms.common.utils.utils import is_xpu_available


class SingleXPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single XPU device."""

    strategy_name = "xpu_single"

    def __init__(
        self,
        device: _DEVICE = "xpu:0",
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):
        if not is_xpu_available():
            raise MisconfigurationException("`SingleXPUStrategy` requires XPU devices to run")

        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def is_distributed(self) -> bool:
        """Returns true if the strategy supports distributed training."""
        return False

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Sets up optimizers."""
        super().setup_optimizers(trainer)
        if len(self.optimizers) != 1:  # type: ignore
            raise RuntimeError("XPU strategy doesn't support multiple optimizers")
        model, optimizer = torch.xpu.optimize(trainer.model, optimizer=self.optimizers[0])  # type: ignore
        self.optimizers = [optimizer]
        trainer.model = model


StrategyRegistry.register(
    SingleXPUStrategy.strategy_name, SingleXPUStrategy, description="Strategy that enables training on single XPU"
)
