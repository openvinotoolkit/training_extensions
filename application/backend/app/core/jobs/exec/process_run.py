# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Process-based job runner for executing jobs in isolated child processes.

This module provides the `ProcessRun` class, which manages the lifecycle of a child process
for job execution, including inter-process communication (IPC) and event translation.
It also includes a factory for creating process runners and the process entrypoint logic.

Classes:
    ProcessRun: Manages a child process for job execution and event streaming.
    ProcessRunnerFactory: Factory for creating `ProcessRun` instances.
Functions:
    _entrypoint: Entrypoint for the child process, executes the job and sends events.
"""

import asyncio
import contextlib
import multiprocessing as mp
import signal
from collections.abc import Iterator
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
from typing import Any

from loguru import logger

from app.core.jobs.models import Done, ExecutionEvent, Failed, Job, JobType, Started
from app.core.logging import LogConfig, job_log_sink, logging_ctx
from app.core.run import ExecutionContext, RunnableFactory, Runner
from app.settings import get_settings

from .exceptions import CancelledExc


def _describe_exit_code(code: int | None) -> str:
    """Build a human readable explanation for a child process exit code.

    A negative exit code means the child was terminated by a signal rather than
    returning normally. In that case no Python-level exception handler ever ran,
    so nothing was written to the job log: the job simply stops mid-way. The most
    common cause is the Linux OOM killer (SIGKILL) reaping a job whose memory
    usage spiked, e.g. during model evaluation on high-resolution inputs.

    Args:
        code: The child process exit code (``Process.exitcode``).

    Returns:
        str: Description of how the process ended.
    """
    if code is None:
        return "process ended without an exit code (it may still be running or was never started)"
    if code >= 0:
        return f"process exit {code}"

    signum = -code
    try:
        signame = signal.Signals(signum).name
    except ValueError:
        signame = f"signal {signum}"

    detail = f"process was terminated by {signame} (exit code {code})"
    # Compare by name: SIGKILL/SIGBUS are not defined on every platform.
    if signame == "SIGKILL":
        detail += (
            ". This is almost always the OS out-of-memory (OOM) killer: the job exceeded the available "
            "memory and was killed before it could report an error. Check dmesg/`journalctl -k` for an "
            "'Out of memory: Killed process' entry, and consider lowering the evaluation batch size"
        )
    elif signame in ("SIGSEGV", "SIGABRT", "SIGBUS"):
        detail += (
            ". This indicates a crash inside a native extension (OpenVINO, PyTorch, OpenCV, ...) rather "
            "than a Python error, so no traceback could be captured"
        )
    return detail


class ProcessRun:
    """
    Manages a child process for job execution and streams domain events to the caller.

    Args:
        ctx (mp.context.SpawnContext): Multiprocessing context for process & IPC.
        runnable_factory (RunnableFactory): Factory to create runnable job instances.
        job (Job): Job specification.
    """

    def __init__(self, ctx: mp.context.SpawnContext, runnable_factory: RunnableFactory, job: Job):
        self._ctx = ctx
        self._runnable_factory = runnable_factory
        self._job = job
        self._parent, self._child = ctx.Pipe(duplex=False)
        self._cancel = ctx.Event()
        self._proc: SpawnProcess | None = None

    def start(self) -> "ProcessRun":
        self._proc = self._ctx.Process(
            target=_entrypoint,
            args=(
                self._runnable_factory,
                self._job.job_type,
                self._job.log_file,
                self._job.params.model_dump_json(),
                self._child,
                self._cancel,
            ),
            name=f"job-{self._job.job_type}-{self._job.id}",
        )
        self._proc.start()
        self._child.close()
        return self

    def events(self) -> Iterator[ExecutionEvent]:
        """Blocking iterator; the control plane decides how to multiplex."""
        try:
            while True:
                msg = self._parent.recv()  # blocks
                yield msg
        except EOFError:
            # Child exited; infer outcome
            code = self._proc.exitcode if self._proc else 1
            if code == 0:
                yield Done()
            else:
                # The child died without sending a Failed event, meaning no Python
                # exception handler ran (killed by a signal, hard native crash, ...).
                # Nothing was written to the job log by the child, so record the
                # cause here - both in the application log and, best effort, in the
                # job's own log file so the user can actually see why it failed.
                details = _describe_exit_code(code)
                self._log_abnormal_exit(details)
                yield Failed(details)
        finally:
            self._parent.close()

    def _log_abnormal_exit(self, details: str) -> None:
        """Record an abnormal child-process termination.

        The child process owns the job log file sink, so when it is killed nothing
        explains the failure to the user. This re-attaches the job log sink from the
        parent process just long enough to append the diagnosis.

        Args:
            details: Human readable description of how the process ended.
        """
        job_name = self._proc.name if self._proc else f"job-{self._job.job_type}-{self._job.id}"
        message = f"Job {self._job.id} ({self._job.job_type}) terminated abnormally: {details}"
        logger.error("{} [{}]", message, job_name)

        try:
            with job_log_sink(LogConfig(log_folder=str(get_settings().job_dir), log_file=self._job.log_file)):
                logger.error(message)
        except Exception:
            logger.exception("Failed to append the abnormal termination reason to the job log file")

    async def stop(self, graceful_timeout: float = 30.0, term_timeout: float = 15.0, kill_timeout: float = 1.0) -> None:
        """
        Stop the process with graceful degradation.

        Args:
            graceful_timeout: How long to wait for graceful shutdown (seconds)
            term_timeout: How long to wait after SIGTERM (seconds)
            kill_timeout: How long to wait after SIGKILL (seconds)
        """
        if self._proc is None:
            return

        # Request graceful shutdown
        self._cancel.set()
        await asyncio.to_thread(self._proc.join, timeout=graceful_timeout)
        if not self._proc.is_alive():
            logger.debug("Process {} stopped gracefully", self._proc.name)
            return

        # Try SIGTERM
        self._proc.terminate()
        await asyncio.to_thread(self._proc.join, timeout=term_timeout)
        if not self._proc.is_alive():
            logger.debug("Process {} terminated gracefully", self._proc.name)
            return

        # Last resort: SIGKILL
        logger.warning("Force killing process {}", self._proc.name)
        self._proc.kill()
        await asyncio.to_thread(self._proc.join, timeout=kill_timeout)
        if self._proc.is_alive():
            logger.error("Process {} doesn't respond to SIGKILL", self._proc.name)


def _entrypoint(
    get_runnable: RunnableFactory, job_type: str, log_file: str, payload: str, conn: Connection, cancel_event: Event
) -> None:
    """
    Entrypoint for the child process.

    Executes the job, sends execution events to the parent process, and handles cancellation.

    Args:
        get_runnable (RunnableFactory): Factory to create runnable job instance.
        job_type (str): Type of job to execute.
        log_file (str): Log file path for job logging.
        payload (str): Serialized job parameters.
        conn (Connection): IPC connection to parent process.
        cancel_event (Event): Event to signal cancellation.
    """
    import traceback

    from app.core.jobs.models import Cancelled, Done, Failed, Progress

    def report(msg: str, p: float, metadata: dict[str, Any] | None = None) -> None:
        if cancel_event.is_set():
            raise CancelledExc
        conn.send(Progress(message=msg, value=p, metadata=metadata))

    runnable = get_runnable(JobType(job_type))

    try:
        conn.send(Started())
        with logging_ctx(LogConfig(log_folder=str(get_settings().job_dir), log_file=log_file)):
            runnable.run(ExecutionContext(payload=payload, report=report))
        conn.send(Done())
    except CancelledExc:
        conn.send(Cancelled())
    except Exception:
        conn.send(Failed(traceback.format_exc()))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


class ProcessRunnerFactory:
    """
    Factory for creating process-based job runners.

    Args:
        runnable_factory (RunnableFactory): Factory to create runnable job instances.

    Methods:
        for_job(job: Job) -> Runner[Job, ExecutionEvent]: Create a ProcessRun instance for the given job.
    """

    def __init__(self, runnable_factory: RunnableFactory) -> None:
        # consider using native context for python 3.14 due to upgrade to 'fork_server' model
        self._ctx = mp.get_context("spawn")
        self._runnable_factory = runnable_factory

    def for_job(self, job: Job) -> Runner[Job, ExecutionEvent]:
        """
        Create a ProcessRun instance for the given job.

        Args:
            job (Job): Job specification.

        Returns:
            Runner[Job, ExecutionEvent]: Process-based job runner.
        """
        return ProcessRun(self._ctx, self._runnable_factory, job)
