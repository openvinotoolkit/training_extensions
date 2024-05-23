# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Scheduler callable to support hyper-parameter optimization (HPO) algorithm."""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from lightning.pytorch.cli import ReduceLROnPlateau
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable


class SchedulerCallableSupportHPO:
    """LR scheduler callable supports OTX hyper-parameter optimization (HPO) algorithm.

    It makes SchedulerCallable pickelable and accessible to parameters.
    It is used for HPO and adaptive batch size.

    Args:
        scheduler_cls: `LRScheduler` class type or string class import path. See examples for details.
        scheduler_kwargs: Keyword arguments used for the initialization of the given `scheduler_cls`.

    Examples:
        This is an example to create `MobileNetV3ForMulticlassCls` with a `StepLR` lr scheduler and
        custom configurations.

        ```python
        from torch.optim.lr_scheduler import StepLR
        from otx.algo.classification.mobilenet_v3_large import MobileNetV3ForMulticlassCls

        model = MobileNetV3ForMulticlassCls(
            num_classes=3,
            scheduler=SchedulerCallableSupportHPO(
                scheduler_cls=StepLR,
                scheduler_kwargs={
                    "step_size": 10,
                    "gamma": 0.5,
                },
            ),
        )
        ```

        It can be created from the string class import path such as

        ```python
        from otx.algo.classification.mobilenet_v3_large import MobileNetV3ForMulticlassCls

        model = MobileNetV3ForMulticlassCls(
            num_classes=3,
            optimizer=SchedulerCallableSupportHPO(
                scheduler_cls="torch.optim.lr_scheduler.StepLR",
                scheduler_kwargs={
                    "step_size": 10,
                    "gamma": 0.5,
                },
            ),
        )
        ```
    """

    def __init__(
        self,
        scheduler_cls: type[LRScheduler] | str,
        scheduler_kwargs: dict[str, int | float | bool | str],
    ):
        if isinstance(scheduler_cls, str):
            splited = scheduler_cls.split(".")
            module_path, class_name = ".".join(splited[:-1]), splited[-1]
            module = importlib.import_module(module_path)

            self.scheduler_init: type[LRScheduler] = getattr(module, class_name)
            self.scheduler_path = scheduler_cls
        elif issubclass(scheduler_cls, LRScheduler | ReduceLROnPlateau):
            self.scheduler_init = scheduler_cls
            self.scheduler_path = scheduler_cls.__module__ + "." + scheduler_cls.__qualname__
        else:
            raise TypeError(scheduler_cls)

        self.scheduler_kwargs = scheduler_kwargs
        self.__dict__.update(scheduler_kwargs)

    def __call__(self, optimizer: Optimizer) -> LRScheduler:
        """Create `torch.optim.LRScheduler` instance for the given parameters."""
        return self.scheduler_init(optimizer, **self.scheduler_kwargs)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return self.__class__, (
            self.scheduler_path,
            self.scheduler_kwargs,
        )

    @classmethod
    def from_callable(cls, func: LRSchedulerCallable) -> SchedulerCallableSupportHPO:
        """Create this class instance from an existing optimizer callable."""
        dummy_params = [nn.Parameter()]
        optimizer = Optimizer(dummy_params, {"lr": 1.0})
        scheduler = func(optimizer)

        allow_names = set(inspect.signature(scheduler.__class__).parameters)

        if isinstance(scheduler, ReduceLROnPlateau):
            # NOTE: Other arguments except "monitor", such as "patience"
            # are not included in the signature of ReduceLROnPlateau.__init__()
            allow_names.update(key for key in inspect.signature(TorchReduceLROnPlateau).parameters)

        block_names = {"optimizer", "last_epoch"}

        scheduler_kwargs = {
            key: value for key, value in scheduler.state_dict().items() if key in allow_names and key not in block_names
        }

        return SchedulerCallableSupportHPO(
            scheduler_cls=scheduler.__class__,
            scheduler_kwargs=scheduler_kwargs,
        )
