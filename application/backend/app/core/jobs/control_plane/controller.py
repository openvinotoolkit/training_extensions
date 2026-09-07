# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import threading

from loguru import logger

from app.core.jobs.models import Cancelled, Done, Failed, Job, JobStatus, JobType, Progress, Started
from app.core.run import Runner, RunnerFactory

from .capacity import Capacity
from .queue import JobQueue


class JobController:
    """
    Asynchronous job orchestration system with capacity management and event-driven control.

    Manages concurrent job execution through an event-driven architecture, coordinating between
    a job queue, runner processes, and capacity constraints. Jobs execute in isolated contexts
    (typically separate processes) to prevent blocking the event loop and contain failures.

    Architecture:
        - Supervisor loop: Continuously polls the job queue for runnable jobs
        - Stale job monitor: Periodically checks for jobs that haven't updated within a threshold
          (default 1 hour) and automatically schedules their termination
        - Capacity control: Enforces maximum parallel job limits via semaphore-based permits
        - Event-driven communication: Jobs emit domain events (Started, Progress, Done, etc.)
          which are pumped from runner threads to the async event loop for state management
        - Cancellation support: Monitors cancellation requests and triggers graceful shutdowns

    The controller maintains separation between CPU-intensive job execution (handled by runners
    in separate processes/threads) and lightweight async orchestration (job lifecycle management,
    event handling, capacity control, stale detection).

    Args:
        jobs_queue: Source of jobs to execute and cancellation state tracker
        runner_factory: Creates appropriate runner contexts for different job types
        max_parallel_jobs: Maximum number of capacity-managed jobs that can execute simultaneously
    """

    def __init__(self, jobs_queue: JobQueue, runner_factory: RunnerFactory, max_parallel_jobs: int) -> None:
        self._jobs_q = jobs_queue
        self._runner_factory = runner_factory
        self._capacity = Capacity(max_parallel_jobs)
        self._running = False
        self._supervisor_task: asyncio.Task | None = None
        self._stale_monitor_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._running = True
        self._supervisor_task = asyncio.create_task(self._supervise_loop(), name="supervisor")
        self._stale_monitor_task = asyncio.create_task(self._monitor_stale_loop(), name="stale_monitor")

    async def stop(self) -> None:
        self._running = False
        loop_tasks = [self._supervisor_task, self._stale_monitor_task]
        for task in loop_tasks:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _supervise_loop(self) -> None:
        while self._running:
            try:
                job = await self._jobs_q.next_runnable()
                logger.info("Starting job with ID: {}", job.id)
                self._start_job(job)
            except Exception:
                logger.exception("Exception during supervise loop")

    async def _monitor_stale_loop(self) -> None:
        """Monitor running jobs and terminate those that haven't updated in 1 hour."""
        stale_threshold = 3600  # 1 hour inactivity threshold
        check_interval = 60  # Check every minute

        while self._running:
            try:
                await asyncio.sleep(check_interval)
                stale_jobs = self._jobs_q.list_stale_jobs(stale_threshold)

                for job in stale_jobs:
                    logger.warning(
                        "Job {} has not updated in {} seconds, marking as stale and scheduling termination",
                        job.id,
                        stale_threshold,
                    )
                    _, result = self._jobs_q.cancel(job.id)
                    logger.info("Cancelling job with ID {} and result {}", job.id, result)
            except Exception:
                logger.exception("Exception during stale job monitoring")

    def _start_job(self, job: Job) -> None:
        task = asyncio.create_task(self._run_job(job))
        self._tasks.add(task)
        task.add_done_callback(lambda t: (self._tasks.discard(t)))

    async def _run_job(self, job: Job) -> None:
        needs_capacity_permit = job.job_type == JobType.TRAIN
        permit_ctx = self._capacity.permit() if needs_capacity_permit else contextlib.nullcontext()
        async with permit_ctx:
            # Check if the job was canceled while waiting for a capacity permit
            if job.status == JobStatus.CANCELLED:
                logger.info("Job {} was canceled while waiting for capacity, skipping execution", job.id)
                return

            job_run = self._runner_factory.for_job(job)
            event_q: asyncio.Queue = asyncio.Queue()

            # Set up event pumping and cancellation monitoring
            cancel_task = self._setup_job_execution(job, job_run, event_q)

            try:
                # Handle events from the job runner
                await self._handle_job_events(job, event_q)
            finally:
                cancel_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_task
                self._jobs_q.cleanup_cancellation_event(job.id)

            # Log when the job has reached its final status
            final_status: JobStatus = job.status
            match final_status:
                case JobStatus.DONE:
                    logger.success("Job completed successfully, job_id: {}", job.id)
                case JobStatus.FAILED:
                    logger.warning("Job failed, job_id: {}", job.id)
                case JobStatus.CANCELLED:
                    logger.info("Job cancelled, job_id: {}", job.id)
                case _:
                    logger.warning("Job ended in unexpected status {}, job_id: {}", final_status.name, job.id)

    def _setup_job_execution(self, job: Job, job_run: Runner, event_q: asyncio.Queue) -> asyncio.Task:
        """Set up event pumping thread and cancellation monitoring task."""
        loop = asyncio.get_running_loop()

        def _pump() -> None:
            for ev in job_run.events():
                asyncio.run_coroutine_threadsafe(event_q.put(ev), loop)
            asyncio.run_coroutine_threadsafe(event_q.put(None), loop)

        async def _cancel() -> None:
            """Watch for job cancellation requests and trigger graceful shutdown."""
            event = self._jobs_q.get_cancellation_event(job.id)
            await event.wait()
            await job_run.stop()

        threading.Thread(target=_pump, daemon=True).start()
        cancel_task = asyncio.create_task(_cancel())
        job_run.start()

        return cancel_task

    @staticmethod
    async def _handle_job_events(job: Job, event_q: asyncio.Queue) -> None:
        """Handle events from the job runner and update job state accordingly."""
        while True:
            evt = await event_q.get()
            if evt is None:
                break

            match evt:
                case Started():
                    job.start()
                case Progress(message=m, value=v, metadata=md):
                    job.advance(percent=v, msg=m, metadata=md)
                case Done():
                    job.finish()
                    break
                case Cancelled():
                    job.cancel()
                    break
                case Failed(details=d):
                    job.fail(msg=d)
                    break
                case _:
                    logger.warning("Unknown trainer event: {}", evt)
