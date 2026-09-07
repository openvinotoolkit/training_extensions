# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import warnings
from collections.abc import Iterator
from queue import Empty, Queue
from typing import Any

from app.core.jobs.models import Cancelled, Done, ExecutionEvent, Failed, Job, Progress, Started
from app.core.run import ExecutionContext, RunnableFactory, Runner

from .exceptions import CancelledExc


class ThreadRun(Runner[Job, ExecutionEvent]):
    """
    Thread-based runner for testing job execution without process overhead.

    Experimental: This class is experimental and intended for development and testing only.

    This runner executes jobs in a separate thread and communicates events through a thread-safe queue, mimicking the
    behavior of ProcessRun but without the overhead of process creation. Designed specifically for integration
    testing where fast execution and easy debugging are prioritized over process isolation.

    Note: This implementation sacrifices process isolation for performance and debuggability.
    Use ProcessRun for production scenarios requiring fault isolation between jobs.
    """

    def __init__(self, get_runnable: RunnableFactory, job: Job):
        warnings.warn(
            "ThreadRun is experimental and intended for development and testing only.", UserWarning, stacklevel=2
        )
        self.get_runnable = get_runnable
        self.job = job
        self._event_queue: Queue[ExecutionEvent | None] = Queue()
        self._cancel_event = threading.Event()
        self._execution_thread: threading.Thread | None = None
        self._started = False

    def start(self) -> "ThreadRun":
        """Start the runner in a separate thread."""
        if self._started:
            return self

        self._started = True
        self._execution_thread = threading.Thread(
            target=self._execute_job, name=f"job-{self.job.job_type}-{self.job.id}", daemon=True
        )
        self._execution_thread.start()
        return self

    def events(self) -> Iterator[ExecutionEvent]:
        """Yield events from the job execution."""
        while True:
            try:
                # Non-blocking get with timeout to allow for graceful shutdown
                event = self._event_queue.get(timeout=0.1)
                if event is None:  # Sentinel value indicating end of events
                    break
                yield event
            except Empty:
                # Continue polling if no events available
                continue

    async def stop(
        self,
        graceful_timeout: float = 6.0,
        term_timeout: float = 3.0,
        kill_timeout: float = 1.0,
        reason: str | None = None,
    ) -> None:
        """Stop the runner by setting the cancellation event.

        Args:
            graceful_timeout: How long to wait for the execution thread to finish.
            term_timeout: Unused - the thread runner cannot escalate beyond cooperative
                cancellation; included for Runner protocol compatibility.
            kill_timeout: Unused - included for Runner protocol compatibility.
            reason: Why the runner is being stopped. Unused here because a thread is
                always cancelled cooperatively and therefore reports its own outcome;
                included for Runner protocol compatibility.
        """
        self._cancel_event.set()

        if self._execution_thread and self._execution_thread.is_alive():
            # Wait for graceful shutdown
            await asyncio.to_thread(self._execution_thread.join, timeout=graceful_timeout)

    def _execute_job(self):
        """Execute the job in a separate thread."""
        try:
            # Always emit Started event first
            self._event_queue.put(Started())

            # Create execution context
            ctx = self._create_execution_context()

            # Execute the runnable
            self.get_runnable(self.job.job_type).run(ctx)

            # If we get here without cancellation, job succeeded
            if not self._cancel_event.is_set():
                self._event_queue.put(Done())

        except CancelledExc:
            self._event_queue.put(Cancelled())
        except Exception as e:
            self._event_queue.put(Failed(str(e)))
        finally:
            # Send sentinel to indicate end of events
            self._event_queue.put(None)

    def _create_execution_context(self) -> ExecutionContext:
        """Create execution context for the thread runnable."""

        class ThreadAwareExecutionContext(ExecutionContext):
            def __init__(self, runner: "ThreadRun"):
                self.runner = runner
                self.report = self._report_impl

            def _report_impl(self, message: str, progress: float, metadata: dict[str, Any] | None = None) -> None:
                if self.runner._cancel_event.is_set():
                    raise CancelledExc("Job cancelled")
                self.runner._event_queue.put(Progress(message, progress, metadata))

        return ThreadAwareExecutionContext(self)


class ThreadRunnerFactory:
    """Factory for creating thread-based job runners."""

    def __init__(self, runnable_factory: RunnableFactory):
        self.runnable_factory = runnable_factory

    def for_job(self, job: Job) -> Runner[Job, ExecutionEvent]:
        """Create a ThreadRun instance for the given job."""
        return ThreadRun(self.runnable_factory, job)
