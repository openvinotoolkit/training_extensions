# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pickle

import pytest
from otx.core.metrics import NullMetricCallable
from otx.core.model.base import DefaultSchedulerCallable, OTXModel
from otx.core.optimizer import OptimizerCallableSupportHPO
from torch import nn
from torch.optim import SGD


class TestOptimizerCallableSupportHPO:
    @pytest.fixture()
    def fxt_params(self):
        model = nn.Linear(10, 10)
        return model.parameters()

    @pytest.fixture(params=["torch.optim.SGD", SGD])
    def fxt_optimizer_cls(self, request):
        return request.param

    @pytest.fixture()
    def fxt_invaliid_optimizer_cls(self):
        class NotOptimizer:
            pass

        return NotOptimizer

    def test_succeed(self, fxt_optimizer_cls, fxt_params):
        optimizer_callable = OptimizerCallableSupportHPO(
            optimizer_cls=fxt_optimizer_cls,
            optimizer_kwargs={
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
        )
        optimizer = optimizer_callable(fxt_params)

        assert isinstance(optimizer, SGD)

        assert optimizer_callable.lr == 0.1
        assert optimizer_callable.momentum == 0.9
        assert optimizer_callable.weight_decay == 1e-4

        assert all(param["lr"] == 0.1 for param in fxt_params)
        assert all(param["momentum"] == 0.9 for param in fxt_params)
        assert all(param["weight_decay"] == 1e-4 for param in fxt_params)

    def test_from_callable(self, fxt_params):
        optimizer_callable = OptimizerCallableSupportHPO.from_callable(
            func=lambda params: SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4),
        )
        optimizer = optimizer_callable(fxt_params)

        assert isinstance(optimizer, SGD)

        assert optimizer_callable.lr == 0.1
        assert optimizer_callable.momentum == 0.9
        assert optimizer_callable.weight_decay == 1e-4

        assert all(param["lr"] == 0.1 for param in fxt_params)
        assert all(param["momentum"] == 0.9 for param in fxt_params)
        assert all(param["weight_decay"] == 1e-4 for param in fxt_params)

    def test_picklable(self, fxt_optimizer_cls):
        optimizer_callable = OptimizerCallableSupportHPO(
            optimizer_cls=fxt_optimizer_cls,
            optimizer_kwargs={
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
        )

        pickled = pickle.dumps(optimizer_callable)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert isinstance(unpickled, OptimizerCallableSupportHPO)
        assert optimizer_callable.optimizer_path == unpickled.optimizer_path
        assert optimizer_callable.optimizer_kwargs == unpickled.optimizer_kwargs

    def test_lazy_instance(self, fxt_optimizer_cls):
        default_optimizer_callable = OptimizerCallableSupportHPO(
            optimizer_cls=fxt_optimizer_cls,
            optimizer_kwargs={
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
        ).to_lazy_instance()

        class _TestOTXModel(OTXModel):
            def __init__(
                self,
                label_info=10,
                optimizer=default_optimizer_callable,
                scheduler=DefaultSchedulerCallable,
                metric=NullMetricCallable,
                torch_compile: bool = False,
            ) -> None:
                super().__init__(label_info, optimizer, scheduler, metric, torch_compile)

            def _create_model(self) -> nn.Module:
                return nn.Linear(10, self.num_classes)

        model = _TestOTXModel()
        optimizers, _ = model.configure_optimizers()
        optimizer = next(iter(optimizers))

        assert isinstance(optimizer, SGD)

    def test_lazy_instance_picklable(self, fxt_optimizer_cls, fxt_params):
        lazy_instance = OptimizerCallableSupportHPO(
            optimizer_cls=fxt_optimizer_cls,
            optimizer_kwargs={
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
        ).to_lazy_instance()

        pickled = pickle.dumps(lazy_instance)
        unpickled = pickle.loads(pickled)  # noqa: S301

        optimizer = unpickled(fxt_params)

        assert isinstance(optimizer, SGD)

        assert unpickled.lr == 0.1
        assert unpickled.momentum == 0.9
        assert unpickled.weight_decay == 1e-4

        assert all(param["lr"] == 0.1 for param in fxt_params)
        assert all(param["momentum"] == 0.9 for param in fxt_params)
        assert all(param["weight_decay"] == 1e-4 for param in fxt_params)
