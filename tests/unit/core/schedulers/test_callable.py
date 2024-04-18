# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pickle

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from otx.core.metrics import NullMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, OTXModel
from otx.core.schedulers import SchedulerCallableSupportHPO
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class TestSchedulerCallableSupportHPO:
    @pytest.fixture()
    def fxt_optimizer(self):
        model = nn.Linear(10, 10)
        return SGD(model.parameters(), lr=1.0)

    @pytest.fixture(
        params=[
            (StepLR, {"step_size": 10, "gamma": 0.5}),
            (CosineAnnealingLR, {"T_max": 10, "eta_min": 0.5}),
            (ReduceLROnPlateau, {"monitor": "my_metric", "patience": 10}),
        ],
        ids=lambda param: param[0].__qualname__,
    )
    def fxt_scheduler_cls_and_kwargs(self, request):
        scheduler_cls, scheduler_kwargs = request.param
        return scheduler_cls, scheduler_kwargs

    def test_succeed(self, fxt_scheduler_cls_and_kwargs, fxt_optimizer):
        scheduler_cls, scheduler_kwargs = fxt_scheduler_cls_and_kwargs
        scheduler_callable = SchedulerCallableSupportHPO(
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
        )
        scheduler = scheduler_callable(fxt_optimizer)

        assert isinstance(scheduler, scheduler_cls)

        for key, value in scheduler_kwargs.items():
            assert getattr(scheduler_callable, key) == value
            assert scheduler_callable.scheduler_kwargs.get(key) == value

            assert scheduler.state_dict().get(key) == value

    def test_from_callable(self, fxt_scheduler_cls_and_kwargs, fxt_optimizer):
        scheduler_cls, scheduler_kwargs = fxt_scheduler_cls_and_kwargs
        scheduler_callable = SchedulerCallableSupportHPO.from_callable(
            func=lambda optimizer: scheduler_cls(optimizer, **scheduler_kwargs),
        )
        scheduler = scheduler_callable(fxt_optimizer)

        assert isinstance(scheduler, scheduler_cls)

        for key, value in scheduler_kwargs.items():
            assert getattr(scheduler_callable, key) == value
            assert scheduler_callable.scheduler_kwargs.get(key) == value

            assert scheduler.state_dict().get(key) == value

    def test_picklable(self, fxt_scheduler_cls_and_kwargs, fxt_optimizer):
        scheduler_cls, scheduler_kwargs = fxt_scheduler_cls_and_kwargs
        scheduler_callable = SchedulerCallableSupportHPO(
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
        )

        pickled = pickle.dumps(scheduler_callable)
        unpickled = pickle.loads(pickled)  # noqa: S301

        scheduler = unpickled(fxt_optimizer)

        assert isinstance(scheduler, scheduler_cls)

        for key, value in scheduler_kwargs.items():
            assert scheduler.state_dict().get(key) == value

    def test_lazy_instance(self, fxt_scheduler_cls_and_kwargs):
        scheduler_cls, scheduler_kwargs = fxt_scheduler_cls_and_kwargs
        default_scheduler_callable = SchedulerCallableSupportHPO(
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
        ).to_lazy_instance()

        class _TestOTXModel(OTXModel):
            def __init__(
                self,
                num_classes=10,
                optimizer=DefaultOptimizerCallable,
                scheduler=default_scheduler_callable,
                metric=NullMetricCallable,
                torch_compile: bool = False,
            ) -> None:
                super().__init__(num_classes, optimizer, scheduler, metric, torch_compile)

            def _create_model(self) -> nn.Module:
                return nn.Linear(10, self.num_classes)

        model = _TestOTXModel()
        _, scheduler_configs = model.configure_optimizers()
        scheduler = next(iter(scheduler_configs))["scheduler"]

        assert isinstance(scheduler, scheduler_cls)

    def test_lazy_instance_picklable(self, fxt_scheduler_cls_and_kwargs, fxt_optimizer):
        scheduler_cls, scheduler_kwargs = fxt_scheduler_cls_and_kwargs
        lazy_instance = SchedulerCallableSupportHPO(
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
        ).to_lazy_instance()

        pickled = pickle.dumps(lazy_instance)
        unpickled = pickle.loads(pickled)  # noqa: S301

        scheduler = unpickled(fxt_optimizer)

        assert isinstance(scheduler, scheduler_cls)

        for key, value in scheduler_kwargs.items():
            assert scheduler.state_dict().get(key) == value
