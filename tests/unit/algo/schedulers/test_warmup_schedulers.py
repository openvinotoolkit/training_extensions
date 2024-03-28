# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from otx.algo.schedulers.warmup_schedulers import LinearWarmupScheduler


def test_linear_warmup_scheduler():
    optimizer = torch.optim.SGD([torch.tensor(0.0)], lr=0.1)
    scheduler = LinearWarmupScheduler(optimizer, num_warmup_steps=1000)

    # Verify initial learning rate
    assert optimizer.param_groups[0]["initial_lr"] == 0.1

    # Perform 500 steps and verify learning rate
    for _ in range(500):
        scheduler.step()
    assert round(optimizer.param_groups[0]["lr"], 2) == 0.05

    # Perform 1000 steps and verify learning rate
    for _ in range(500):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.1

    # Perform 1500 steps and verify learning rate
    for _ in range(500):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.1

    # Perform 2000 steps and verify learning rate
    for _ in range(500):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.1
