# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Warm-up schedulers for the OTX2.0."""
from __future__ import annotations

import torch


class LinearWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear Warmup scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 1000,
        interval: str = "step",
    ):
        if not num_warmup_steps > 0:
            msg = f"num_warmup_steps should be > 0, got {num_warmup_steps}"
            raise ValueError(msg)
        self.num_warmup_steps = num_warmup_steps
        self.interval = interval
        super().__init__(optimizer, lambda step: min(step / num_warmup_steps, 1.0))
