# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from enum import StrEnum

from app.core.run import ExecutionContext, Runnable


class RunnableBehaviour(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SLOW = "slow"
    INSTANT = "instant"
    FAILURE_WHILE_CANCELLING = "failure_while_cancelling"


class MockRunnable(Runnable):
    """
    Mock runnable implementation for integration testing of job execution.

    Behaviors:
    - SUCCESS: Normal execution with configurable progress steps and timing
    - FAILURE: Simulates job failure by raising ValueError after initial progress
    - SLOW: Extended execution with many small progress increments for cancellation testing
    - INSTANT: Very fast execution with minimal progress steps
    - FAILURE_WHILE_CANCELLING: Ignores cancellation and raises ValueError regardless, simulating a runner
      being force-terminated (e.g. SIGTERM/SIGKILL) instead of reacting to the cancellation request
    """

    def __init__(
        self,
        behavior: RunnableBehaviour = RunnableBehaviour.SUCCESS,
        progress_steps: list | None = None,
        execution_time: float = 0.1,
    ):
        self.behavior = behavior
        self.progress_steps = progress_steps or [25.0, 50.0, 75.0, 100.0]
        self.execution_time = execution_time  # Time between progress steps

    def run(self, ctx: ExecutionContext) -> None:
        """Execute the runnable with given context."""
        match self.behavior:
            case RunnableBehaviour.SUCCESS:
                for progress in self.progress_steps:
                    ctx.report("", progress, None)
                    if progress < 100.0:  # Don't sleep after final progress
                        time.sleep(self.execution_time)

            case RunnableBehaviour.FAILURE:
                ctx.report("", 10.0, None)
                time.sleep(self.execution_time)
                raise ValueError("Mock failure")

            case RunnableBehaviour.SLOW:
                for i in range(20):  # More steps for better cancellation testing
                    progress = i * 5  # 0, 5, 10, ... 95
                    ctx.report("", progress, None)
                    time.sleep(self.execution_time)

            case RunnableBehaviour.INSTANT:
                # Very fast execution
                for progress in [50.0, 100.0]:
                    ctx.report("", progress, None)

            case RunnableBehaviour.FAILURE_WHILE_CANCELLING:
                # Never checks for cancellation, so it fails instead of reacting to the cancel signal
                time.sleep(self.execution_time)
                raise ValueError("Mock failure while cancelling")

            case _:
                raise ValueError(f"Unknown behavior: {self.behavior}")
