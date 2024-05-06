# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from otx.core.schedulers.warmup_schedulers import LinearWarmupScheduler, LinearWarmupSchedulerCallable
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.sgd import SGD


@pytest.fixture()
def fxt_optimizer():
    model = nn.Linear(10, 10)
    return SGD(params=model.parameters(), lr=1.0)


class TestLinearWarmupScheduler:
    def test_activation(self, fxt_optimizer):
        num_warmup_steps = 3
        scheduler = LinearWarmupScheduler(optimizer=fxt_optimizer, num_warmup_steps=num_warmup_steps)

        for _ in range(num_warmup_steps):
            assert scheduler.activated
            scheduler.step()

        assert not scheduler.activated


class TestLinearWarmupSchedulerCallable:
    def test_num_warmup_steps(self, fxt_optimizer):
        # No linear warmup scheduler because num_warmup_steps = 0 by default
        scheduler_callable = LinearWarmupSchedulerCallable(
            main_scheduler_callable=lambda optimizer: StepLR(optimizer, step_size=10, gamma=0.5),
        )

        schedulers = scheduler_callable(fxt_optimizer)
        assert len(schedulers) == 1
        assert isinstance(schedulers[0], StepLR)

        # linear warmup scheduler exists because num_warmup_steps > 0
        scheduler_callable = LinearWarmupSchedulerCallable(
            main_scheduler_callable=lambda optimizer: StepLR(optimizer, step_size=10, gamma=0.5),
            num_warmup_steps=10,
            warmup_interval="epoch",
        )

        schedulers = scheduler_callable(fxt_optimizer)

        assert len(schedulers) == 2
        assert isinstance(schedulers[0], StepLR)
        assert isinstance(schedulers[1], LinearWarmupScheduler)
        assert schedulers[1].num_warmup_steps == 10
        assert schedulers[1].interval == "epoch"

    def test_monitor(self, fxt_optimizer):
        # If monitor None, do not override monitor.
        scheduler_callable = LinearWarmupSchedulerCallable(
            main_scheduler_callable=lambda optimizer: ReduceLROnPlateau(optimizer, monitor="not_my_metric"),
            num_warmup_steps=10,
            monitor=None,
        )
        schedulers = scheduler_callable(fxt_optimizer)

        assert len(schedulers) == 2
        assert schedulers[0].monitor == "not_my_metric"
        assert isinstance(schedulers[1], LinearWarmupScheduler)

        # Set monitor from "not_my_metric" to "my_metric"
        scheduler_callable = LinearWarmupSchedulerCallable(
            main_scheduler_callable=lambda optimizer: ReduceLROnPlateau(optimizer, monitor="not_my_metric"),
            num_warmup_steps=10,
            monitor="my_metric",
        )

        schedulers = scheduler_callable(fxt_optimizer)

        assert len(schedulers) == 2
        assert schedulers[0].monitor == "my_metric"
        assert isinstance(schedulers[1], LinearWarmupScheduler)
