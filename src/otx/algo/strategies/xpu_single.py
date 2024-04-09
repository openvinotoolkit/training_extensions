"""Lightning strategy for single XPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning.pytorch.strategies import StrategyRegistry
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from otx.utils.utils import is_xpu_available

if TYPE_CHECKING:
    import lightning.pytorch as pl
    from lightning.pytorch.plugins.precision import PrecisionPlugin
    from lightning_fabric.plugins import CheckpointIO
    from lightning_fabric.utilities.types import _DEVICE


class SingleXPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single XPU device."""

    strategy_name = "xpu_single"

    def __init__(
        self,
        device: _DEVICE = "xpu:0",
        accelerator: pl.accelerators.Accelerator | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: PrecisionPlugin | None = None,
    ):
        if not is_xpu_available():
            msg = "`SingleXPUStrategy` requires XPU devices to run"
            raise MisconfigurationException(msg)

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

    def setup_optimizers(self, trainer: pl.Trainer) -> None:
        """Sets up optimizers."""
        super().setup_optimizers(trainer)
        if len(self.optimizers) > 1:  # type: ignore[has-type]
            msg = "XPU strategy doesn't support multiple optimizers"
            raise RuntimeError(msg)
        if trainer.task != "SEMANTIC_SEGMENTATION":
            if len(self.optimizers) == 1:  # type: ignore[has-type]
                model, optimizer = torch.xpu.optimize(trainer.model, optimizer=self.optimizers[0])  # type: ignore[has-type]
                self.optimizers = [optimizer]
                self.model = model
            else:  # for inference
                trainer.model.eval()
                self.model = torch.xpu.optimize(trainer.model)


StrategyRegistry.register(
    SingleXPUStrategy.strategy_name,
    SingleXPUStrategy,
    description="Strategy that enables training on single XPU",
)
