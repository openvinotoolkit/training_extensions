"""Lightning strategy for single XPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.strategies import StrategyRegistry
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from otx.utils.device import is_xpu_available

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


StrategyRegistry.register(
    SingleXPUStrategy.strategy_name,
    SingleXPUStrategy,
    description="Strategy that enables training on single XPU",
)
