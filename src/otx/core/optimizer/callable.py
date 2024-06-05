# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Optimizer callable to support hyper-parameter optimization (HPO) algorithm."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from torch import nn
from torch.optim.optimizer import Optimizer

if TYPE_CHECKING:
    from lightning.pytorch.cli import OptimizerCallable
    from torch.optim.optimizer import params_t


class OptimizerCallableSupportHPO:
    """Optimizer callable supports OTX hyper-parameter optimization (HPO) algorithm.

    It makes OptimizerCallable pickelable and accessible to parameters.
    It is used for HPO and adaptive batch size.

    Args:
        optimizer_cls: Optimizer class type or string class import path. See examples for details.
        optimizer_kwargs: Keyword arguments used for the initialization of the given `optimizer_cls`.

    Examples:
        This is an example to create `MobileNetV3ForMulticlassCls` with a `SGD` optimizer and
        custom configurations.

        ```python
        from torch.optim import SGD
        from otx.algo.classification.mobilenet_v3_large import MobileNetV3ForMulticlassCls

        model = MobileNetV3ForMulticlassCls(
            num_classes=3,
            optimizer=OptimizerCallableSupportHPO(
                optimizer_cls=SGD,
                optimizer_kwargs={
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                },
            ),
        )
        ```

        It can be created from the string class import path such as

        ```python
        from otx.algo.classification.mobilenet_v3_large import MobileNetV3ForMulticlassCls

        model = MobileNetV3ForMulticlassCls(
            num_classes=3,
            optimizer=OptimizerCallableSupportHPO(
                optimizer_cls="torch.optim.SGD",
                optimizer_kwargs={
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                },
            ),
        )
        ```
    """

    def __init__(
        self,
        optimizer_cls: type[Optimizer] | str,
        optimizer_kwargs: dict[str, int | float | bool],
    ):
        if isinstance(optimizer_cls, str):
            splited = optimizer_cls.split(".")
            module_path, class_name = ".".join(splited[:-1]), splited[-1]
            module = importlib.import_module(module_path)

            self.optimizer_init: type[Optimizer] = getattr(module, class_name)
            self.optimizer_path = optimizer_cls
        elif issubclass(optimizer_cls, Optimizer):
            self.optimizer_init = optimizer_cls
            self.optimizer_path = optimizer_cls.__module__ + "." + optimizer_cls.__qualname__
        else:
            raise TypeError(optimizer_cls)

        self.optimizer_kwargs = optimizer_kwargs
        self.__dict__.update(optimizer_kwargs)

    def __call__(self, params: params_t) -> Optimizer:
        """Create `torch.optim.Optimizer` instance for the given parameters."""
        return self.optimizer_init(params, **self.optimizer_kwargs)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return self.__class__, (
            self.optimizer_path,
            self.optimizer_kwargs,
        )

    @classmethod
    def from_callable(cls, func: OptimizerCallable) -> OptimizerCallableSupportHPO:
        """Create this class instance from an existing optimizer callable."""
        dummy_params = [nn.Parameter()]
        optimizer = func(dummy_params)

        param_group = next(iter(optimizer.param_groups))

        return OptimizerCallableSupportHPO(
            optimizer_cls=optimizer.__class__,
            optimizer_kwargs={key: value for key, value in param_group.items() if key != "params"},
        )
