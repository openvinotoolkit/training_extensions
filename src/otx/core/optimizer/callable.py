# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Optimizer callable to support hyper-parameter optimization (HPO) algorithm."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Sequence

from torch import nn
from torch.optim.optimizer import Optimizer

from otx.core.utils.jsonargparse import ClassType, lazy_instance

if TYPE_CHECKING:
    from lightning.pytorch.cli import OptimizerCallable
    from torch.optim.optimizer import params_t


class OptimizerCallableSupportHPO:
    """Optimizer callable supports OTX hyper-parameter optimization (HPO) algorithm.

    Args:
        optimizer_cls: Optimizer class type or string class import path. See examples for details.
        optimizer_kwargs: Keyword arguments used for the initialization of the given `optimizer_cls`.
        search_hparams: Sequence of optimizer hyperparameter names which can be tuned by the OTX HPO algorithm.

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
        search_hparams: Sequence[str] = ("lr",),
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

        for search_hparam in search_hparams:
            if search_hparam not in optimizer_kwargs:
                msg = (
                    f"Search hyperparamter={search_hparam} should be existed in "
                    f"optimizer keyword arguments={optimizer_kwargs} as well."
                )
                raise ValueError(msg)

        self.search_hparams = list(search_hparams)
        self.optimizer_kwargs = optimizer_kwargs
        self.__dict__.update(optimizer_kwargs)

    def __call__(self, params: params_t) -> Optimizer:
        """Create `torch.optim.Optimizer` instance for the given parameters."""
        return self.optimizer_init(params, **self.optimizer_kwargs)

    def to_lazy_instance(self) -> ClassType:
        """Return lazy instance of this class.

        Because OTX is rely on jsonargparse library,
        the default value of class initialization
        argument should be the lazy instance.
        Please refer to https://jsonargparse.readthedocs.io/en/stable/#default-values
        for more details.

        Examples:
            This is an example to implement a new model with a `SGD` optimizer and
            custom configurations as a default.

            ```python
            class MyAwesomeMulticlassClsModel(OTXMulticlassClsModel):
                def __init__(
                    self,
                    num_classes: int,
                    optimizer: OptimizerCallable = OptimizerCallableSupportHPO(
                        optimizer_cls=SGD,
                        optimizer_kwargs={
                            "lr": 0.1,
                            "momentum": 0.9,
                            "weight_decay": 1e-4,
                        },
                    ).to_lazy_instance(),
                    scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
                    metric: MetricCallable = MultiClassClsMetricCallable,
                    torch_compile: bool = False,
                ) -> None:
                ...
            ```
        """
        return lazy_instance(
            OptimizerCallableSupportHPO,
            optimizer_cls=self.optimizer_path,
            optimizer_kwargs=self.optimizer_kwargs,
            search_hparams=self.search_hparams,
        )

    def __reduce__(self) -> str | tuple[Any, ...]:
        return self.__class__, (
            self.optimizer_path,
            self.optimizer_kwargs,
            self.search_hparams,
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
