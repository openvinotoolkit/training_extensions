# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle management"""

import json
import multiprocessing as mp
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from multiprocessing.synchronize import Condition
from pathlib import Path

from aiortc import RTCConfiguration, RTCIceServer
from fastapi import FastAPI
from loguru import logger

from app.core.jobs import JobController, JobQueue, ProcessRunnerFactory
from app.core.jobs.models import JobType
from app.core.logging import LogConfig, setup_logging
from app.core.run import Runnable, RunnableFactory
from app.db import MigrationFatalError, MigrationManager, get_db_session
from app.execution.builders import (
    build_export_dataset,
    build_import_as_new_project,
    build_import_to_project,
    build_prepare_dataset,
    build_quantizer,
    build_trainer,
)
from app.scheduler import Scheduler
from app.services import (
    DatasetRevisionService,
    DatasetService,
    LabelService,
    MediaService,
    ModelService,
    PipelineService,
    ProjectService,
    TrainingConfigurationService,
)
from app.services.base_weights_service import BaseWeightsService
from app.services.data_collect import DataCollector
from app.services.event.event_bus import EventBus
from app.services.inference import InferenceServer
from app.services.subset_assignment import SubsetAssigner, SubsetService
from app.services.video import CacheConfig, VideoService
from app.settings import get_settings
from app.webrtc import SDPHandler, WebRTCManager, WebRTCSettings

# Dedicated process exit code for fatal, non-restartable migration failures.
# A failed or incompatible schema migration will deterministically fail again,
# so the launcher/supervisor must NOT restart the process when it sees this code.
MIGRATION_FATAL_EXIT_CODE = 3

# Name of the machine-readable status file the backend drops into DATA_DIR right
# before exiting with MIGRATION_FATAL_EXIT_CODE. The Tauri shell reads this file when it sees the
# fatal exit code, presents the recovery guidance to the user, then deletes it.
FATAL_STATUS_FILENAME = "fatal_status.json"


def write_fatal_status(data_dir: Path, *, reason: str, backup_path: Path | None, database_path: Path) -> None:
    """Persist a structured description of a fatal startup failure for the UI shell.

    Written to ``<data_dir>/<FATAL_STATUS_FILENAME>`` as JSON so a front-end
    process (the Tauri side-car supervisor) can surface the exact recovery
    details — most importantly the backup file location — even when the backend
    has no console (e.g. Windows release runs with CREATE_NO_WINDOW).

    Failures to write are logged but never raised: the status file is a
    best-effort convenience on top of the log output, not a hard dependency.

    Args:
        data_dir: Directory shared with the UI shell (the backend's DATA_DIR).
        reason: Short machine-readable failure category, e.g. ``"migration"``.
        backup_path: Location of the pre-migration backup, or ``None`` if none exists.
        database_path: Path of the database file the backup should be restored to.
    """
    status_path = data_dir / FATAL_STATUS_FILENAME
    payload = {
        "fatal": reason,
        "backup_path": str(backup_path) if backup_path is not None else None,
        "database_path": str(database_path),
    }
    try:
        status_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Wrote fatal status file for the UI shell: {}", status_path)
    except OSError as exc:
        logger.warning("Could not write fatal status file {}: {}", status_path, exc)


def clear_fatal_status(data_dir: Path) -> None:
    """Remove any stale fatal-status file from a previous run.

    Called on a successful startup so a leftover file (e.g. if the UI shell
    exited before consuming it) can never trigger a spurious recovery dialog on
    a later, healthy launch.
    """
    status_path = data_dir / FATAL_STATUS_FILENAME
    try:
        status_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove stale fatal status file {}: {}", status_path, exc)


def setup_job_controller(
    data_dir: Path, staged_datasets_dir: Path | None, max_parallel_jobs: int
) -> tuple[JobQueue, JobController]:
    """
    Initializes and configures the job queue and job controller for managing parallel job execution.

    Sets up the infrastructure to run jobs concurrently and registers classes that comply with the Runnable protocol,
    each associated with a job type and its required dependencies. These classes are executed in a context defined
    by the runner factory.

    Args:
        data_dir: Path to the directory containing data required for job execution.
        staged_datasets_dir: Path to the directory for storing staged datasets.
        max_parallel_jobs (int): Maximum number of jobs that can run concurrently.

    Returns:
        tuple[JobQueue, JobController]: The job queue and the configured job controller.
    """
    if not staged_datasets_dir:
        raise ValueError("staged_datasets_dir must be provided")
    q = JobQueue()
    job_runnable_factory = RunnableFactory[JobType, Runnable]()
    label_service = LabelService()
    dataset_service = DatasetService(
        label_service=label_service,
        media_service=MediaService(data_dir=data_dir),
    )
    project_service = ProjectService(
        data_dir=data_dir,
        label_service=label_service,
        pipeline_service=PipelineService(),
    )
    dataset_revision_service = DatasetRevisionService(data_dir=data_dir)
    job_runnable_factory.register(
        JobType.TRAIN,
        partial(
            build_trainer,
            base_weights_service=BaseWeightsService(data_dir=data_dir),
            subset_service=SubsetService(),
            subset_assigner=SubsetAssigner(),
            dataset_service=dataset_service,
            dataset_revision_service=dataset_revision_service,
            model_service=ModelService(data_dir=data_dir),
            training_configuration_service=TrainingConfigurationService(),
            data_dir=data_dir,
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.QUANTIZE,
        partial(
            build_quantizer,
            data_dir=data_dir,
            model_service=ModelService(data_dir=data_dir),
            dataset_revision_service=dataset_revision_service,
            project_service=project_service,
            training_configuration_service=TrainingConfigurationService(),
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.EXPORT_DATASET,
        partial(
            build_export_dataset,
            staged_datasets_dir=staged_datasets_dir,
            dataset_service=dataset_service,
            dataset_revision_service=dataset_revision_service,
            project_service=project_service,
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.PREPARE_DATASET_FOR_IMPORT,
        partial(
            build_prepare_dataset,
            staged_datasets_dir=staged_datasets_dir,
        ),
    )
    job_runnable_factory.register(
        JobType.IMPORT_DATASET_TO_PROJECT,
        partial(
            build_import_to_project,
            staged_datasets_dir=staged_datasets_dir,
            dataset_service=dataset_service,
            label_service=label_service,
            media_service=MediaService(data_dir=data_dir),
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.IMPORT_DATASET_AS_NEW_PROJECT,
        partial(
            build_import_as_new_project,
            staged_datasets_dir=staged_datasets_dir,
            project_service=project_service,
            dataset_service=dataset_service,
            label_service=label_service,
            media_service=MediaService(data_dir=data_dir),
            db_session_factory=get_db_session,
        ),
    )
    process_runner_factory = ProcessRunnerFactory(job_runnable_factory)
    job_controller = JobController(
        jobs_queue=q, runner_factory=process_runner_factory, max_parallel_jobs=max_parallel_jobs
    )
    return q, job_controller


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # noqa: PLR0915
    """FastAPI lifespan context manager"""
    # Startup
    settings = get_settings()
    settings.ensure_dirs_exist()
    app.state.settings = settings

    # Setup logging
    setup_logging(config=LogConfig(level=settings.log_level))

    # Initialize database
    migration_manager = MigrationManager(settings)
    try:
        initialized = migration_manager.initialize_database()
    except MigrationFatalError as e:
        # A failed/incompatible database migration is fatal and non-restartable.
        # Log the traceback and recovery guidance, then terminate the process with a
        # dedicated exit code. os._exit is used (instead of raising) so the exit code
        # reliably reaches the launcher/supervisor rather than being swallowed by the
        # ASGI server's lifespan error handling.
        logger.exception("Fatal database migration error; exiting without restart: {}", e)
        if e.backup_path is not None:
            logger.error(
                "To recover, restore the pre-migration database backup: rename the backup file "
                "'{}' back to '{}' (the original database file), overwriting the partially "
                "migrated database.",
                e.backup_path,
                migration_manager.database_path,
            )
            logger.error("After restoring the backup, downgrade the application to the previous version.")
        else:
            logger.error(
                "No pre-migration database backup is available. To recover, downgrade the "
                "application to the previous version."
            )
        logger.error("If the problem persists, please create a ticket on https://github.com/open-edge-platform/geti.")
        # Drop a machine-readable status file next to the database so the
        # UI shell can show the exact backup path to the user
        write_fatal_status(
            settings.data_dir,
            reason="migration",
            backup_path=e.backup_path,
            database_path=migration_manager.database_path,
        )
        # loguru is configured with enqueue=True (see setup_logging), so records are emitted
        # from a background thread. os._exit() terminates immediately and would drop any
        # records still on the queue, so drain the sinks before exiting.
        await logger.complete()
        os._exit(MIGRATION_FATAL_EXIT_CODE)

    if not initialized:
        logger.error("Failed to initialize database. Application cannot start.")
        raise RuntimeError("Database initialization failed")

    # Startup succeeded: clear any stale fatal-status file from a previous failed run so
    # it can't be mistaken for a fresh failure by the UI shell.
    clear_fatal_status(settings.data_dir)

    # Worker processes are created with the "spawn" method to ensure a clean state and avoid issues with shared
    # resources, especially when the workers involve GPU usage or complex libraries that may not be fork-safe.
    # See https://github.com/open-edge-platform/training_extensions/issues/5701 for more details.
    mp_ctx = mp.get_context("spawn")

    # Condition to notify processes about source updates
    source_changed_condition: Condition = mp_ctx.Condition()
    # Event to signal that the model has to be reloaded
    model_reload_event = mp_ctx.Event()
    # Event to signal that the inference parameters of the loaded model changed (no reload needed)
    inference_params_event = mp_ctx.Event()

    event_bus = EventBus(
        source_changed_condition=source_changed_condition,
        model_reload_event=model_reload_event,
        inference_params_event=inference_params_event,
    )
    app.state.event_bus = event_bus

    cache_config = CacheConfig(
        ttl=settings.video_cache_ttl,
        cleanup_interval=settings.video_cache_cleanup_interval,
    )
    video_service = VideoService(cache_config=cache_config)
    app.state.video_service = video_service

    data_collector = DataCollector(data_dir=settings.data_dir, event_bus=event_bus)
    app.state.data_collector = data_collector

    inference_server = InferenceServer(data_dir=settings.data_dir)
    app.state.inference_server = inference_server

    # Initialize Scheduler
    app_scheduler = Scheduler(
        event_bus=event_bus, data_collector=data_collector, inference_server=inference_server, mp_ctx=mp_ctx
    )
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler

    webrtc_settings = WebRTCSettings(
        config=RTCConfiguration(iceServers=[RTCIceServer(**server) for server in settings.ice_servers]),
        advertise_ip=settings.webrtc_advertise_ip,
    )
    sdp_handler = SDPHandler()
    webrtc_manager = WebRTCManager(app_scheduler.rtc_stream_broadcaster, webrtc_settings, sdp_handler)
    app.state.webrtc_manager = webrtc_manager
    logger.info("Application startup completed")

    job_queue, job_controller = setup_job_controller(
        data_dir=settings.data_dir,
        staged_datasets_dir=settings.staged_datasets_dir,
        max_parallel_jobs=settings.gpu_slots,
    )
    app.state.job_queue = job_queue

    await job_controller.start()

    yield

    await job_controller.stop()
    # Shutdown
    logger.info("Shutting down {} application...", settings.app_name)
    video_service.close()
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
