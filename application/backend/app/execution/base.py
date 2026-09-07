# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, Generic, TypeVar

from loguru import logger

from app.core.jobs.models import JobParams
from app.core.run import ExecutionContext, Runnable

T = TypeVar("T")


def step(name: str, complete: float = 0.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to mark a method as a step of the Execution implementation.

    It expects the decorated method to be part of an Execution subclass. The decorator adds progress reporting around
    the execution of the step.

    Usage:
        class Training(Execution):
            @step("Prepare Weights")
            def prepare_weights(self, ...) -> None:
                # implementation
                pass

    Args:
        name: Human-readable name for the step (used in logging and progress reporting).
        complete: Optional float indicating the completion percentage after this step.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self: "Execution", *args: Any, **kwargs: Any) -> T:
            self.update_message(f"Started: {name}")
            try:
                result = func(self, *args, **kwargs)
            except ExecutionErr as e:
                self.update_message(str(e), level="ERROR")
                raise
            except Exception:
                self.update_message(f"Failed: {name}", level="ERROR")
                raise
            if not self._pinned_message:
                self._report_progress(f"Completed: {name}", percent=complete)
            else:
                self._pinned_message = False  # reset pinned message state for the next step
                logger.info(f"Completed: {name}")
            return result

        return wrapper

    return decorator


class ExecutionErr(Exception):
    """Raised when an execution step fails in an expected, user-facing way."""


JobParamsT = TypeVar("JobParamsT", bound=JobParams)


class Execution(Runnable, ABC, Generic[JobParamsT]):
    """
    Abstract base class for Runnable implementations.

    Subclasses should implement their logic by defining methods decorated with @step.
    """

    params_type: type[JobParamsT]

    def __init__(self) -> None:
        self._ctx: ExecutionContext | None = None
        self._pinned_message: bool = False

    def parse_params(self, ctx: ExecutionContext) -> JobParamsT:
        """Parse and validate parameters from execution context."""
        return self.params_type.model_validate_json(ctx.payload)

    @abstractmethod
    def execute(self, params: JobParamsT) -> None:
        """Execute the main logic using parsed params."""
        ...

    def run(self, ctx: ExecutionContext) -> None:
        """Template method that handles context setup and param parsing."""
        self._ctx = ctx
        self.execute(self.parse_params(ctx))

    def __report_to_context(self, msg: str, percent: float, metadata: dict[str, Any] | None = None) -> None:
        """Report progress to execution context if available."""
        if self._ctx is not None:
            self._ctx.report(msg, percent, metadata)

    def update_message(self, msg: str, level: str = "INFO") -> None:
        """Update the current progress message without changing the percentage."""
        self._report_progress(msg=msg, level=level)

    def pin_message(self, msg: str, level: str = "WARNING") -> None:
        """Update the message and prevent the step decorator from overriding it on completion."""
        self._pinned_message = True
        self._report_progress(msg=msg, level=level)

    def update_progress(self, percent: float) -> None:
        """Update the current progress percentage without changing the message."""
        if percent <= 0.0 or percent > 100.0:
            raise ValueError(f"Progress percentage must be in (0; 100], got {percent}")
        self._report_progress(percent=percent)

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update the current progress metadata without changing the message or percentage."""
        self._report_progress(metadata=metadata)

    def heartbeat(self) -> None:
        """Signal that the job is still alive, without changing the message or percentage.

        A step only reports progress when it starts and when it finishes. Any step that
        runs longer than the control plane's stale-job threshold without reporting is
        therefore misclassified as hung and gets terminated with SIGTERM/SIGKILL -- which
        kills the process without running any Python exception handler, so the job log
        simply stops mid-way with no error.

        Long-running, non-incremental work (model evaluation in particular, which runs on
        CPU for the OpenVINO and ONNX variants and can take hours) must call this
        periodically. It also makes such work cooperatively cancellable, because the
        reporting hook raises when a cancellation has been requested.
        """
        self._report_progress()

    def _report_progress(
        self, msg: str = "", percent: float = 0.0, metadata: dict[str, Any] | None = None, level: str = "INFO"
    ) -> None:
        if msg:
            logger.opt(exception=level == "ERROR").log(level, msg)
        self.__report_to_context(msg=msg, percent=percent, metadata=metadata)
