# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, Body, Depends, HTTPException, status
from loguru import logger
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.api.dependencies import (
    get_data_dir,
    get_dataset_view_service,
    get_job_dir,
    get_job_queue,
    get_project_service,
    get_system_service,
)
from app.api.schemas.jobs import JobRequest, JobType, JobView
from app.api.validators import JobID
from app.core.jobs.control_plane import CancellationResult, JobQueue
from app.core.jobs.models import JobStatus
from app.models import (
    DatasetFormat,
    ExportDatasetJob,
    ExportDatasetJobParams,
    QuantizationJob,
    QuantizationJobParams,
    TrainingJob,
    TrainingJobParams,
)
from app.models.jobs import (
    ImportDatasetAsNewProjectJob,
    ImportDatasetAsNewProjectJobParams,
    ImportDatasetToProjectJob,
    ImportDatasetToProjectJobParams,
    PrepareDatasetForImportJob,
    PrepareDatasetForImportJobParams,
)
from app.services import DatasetViewService, ProjectService, SystemService
from app.services.model_manifest_service import ModelManifestService

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobView,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Job successfully created"},
        status.HTTP_400_BAD_REQUEST: {"description": "Unknown job type or invalid parameters"},
        status.HTTP_404_NOT_FOUND: {"description": "Project, dataset or dataset view not found"},
        status.HTTP_409_CONFLICT: {"description": "Dataset is locked by another job"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Dataset already in datumaro format and ready for import"
        },
    },
)
async def submit_job(
    job_request: Annotated[JobRequest, Body()],
    job_queue: Annotated[JobQueue, Depends(get_job_queue)],
    job_dir: Annotated[Path, Depends(get_job_dir)],
    data_dir: Annotated[Path, Depends(get_data_dir)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    system_service: Annotated[SystemService, Depends(get_system_service)],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> JobView:
    """Create a new job and submit it to the scheduler."""
    try:
        job = None
        job_id = uuid4()
        match job_request.job_type:
            case JobType.TRAIN:
                device = system_service.training_device(job_request.parameters.device)
                project = project_service.get_project_by_id(job_request.project_id)
                arch_id = job_request.parameters.model_architecture_id
                arch_name = ModelManifestService.get_model_manifest_by_id(arch_id).name
                job = TrainingJob(
                    id=job_id,
                    project_id=project.id,
                    log_dir=job_dir,
                    data_dir=data_dir,
                    params=TrainingJobParams(
                        device=device,
                        model_architecture_id=arch_id,
                        model_architecture_name=arch_name,
                        parent_model_revision_id=job_request.parameters.parent_model_revision_id,
                        task=project.task,
                        project_id=project.id,
                        job_id=job_id,
                        dataset_revision_id=job_request.parameters.dataset_revision_id,
                    ),
                )
            case JobType.QUANTIZE:
                project = project_service.get_project_by_id(job_request.project_id)
                arch_id = job_request.parameters.model_architecture_id
                arch_name = ModelManifestService.get_model_manifest_by_id(arch_id).name
                job = QuantizationJob(
                    id=job_id,
                    project_id=project.id,
                    log_dir=job_dir,
                    data_dir=data_dir,
                    params=QuantizationJobParams(
                        job_id=job_id,
                        project_id=project.id,
                        model_id=job_request.parameters.model_id,
                        model_architecture_id=arch_id,
                        model_architecture_name=arch_name,
                        max_calibration_subset_size=job_request.parameters.max_calibration_subset_size,
                        max_drop=job_request.parameters.max_drop,
                        max_num_iterations=job_request.parameters.max_num_iterations,
                    ),
                )
            case JobType.PREPARE_DATASET_FOR_IMPORT:
                job = PrepareDatasetForImportJob(
                    id=job_id,
                    params=PrepareDatasetForImportJobParams(
                        staged_dataset_id=job_request.staged_dataset_id,
                    ),
                )
            case JobType.IMPORT_DATASET_AS_NEW_PROJECT:
                job = ImportDatasetAsNewProjectJob(
                    id=job_id,
                    params=ImportDatasetAsNewProjectJobParams(
                        staged_dataset_id=job_request.staged_dataset_id,
                        project_name=job_request.parameters.project.name,
                        task_type=job_request.parameters.project.task_type,
                        labels=job_request.parameters.filters.labels,
                        subsets=job_request.parameters.filters.subsets,
                        include_unannotated=job_request.parameters.filters.include_unannotated,
                    ),
                )
            case JobType.IMPORT_DATASET_TO_PROJECT:
                project = project_service.get_project_by_id(job_request.project_id)
                job = ImportDatasetToProjectJob(
                    id=job_id,
                    project_id=project.id,
                    params=ImportDatasetToProjectJobParams(
                        staged_dataset_id=job_request.staged_dataset_id,
                        project_id=project.id,
                        task=project.task,
                        labels_mapping=job_request.parameters.labels_mapping,
                        include_unannotated=job_request.parameters.include_unannotated,
                    ),
                )
            case JobType.EXPORT_DATASET:
                project = project_service.get_project_by_id(job_request.project_id)
                if job_request.dataset_view_id is not None:
                    # Raises ResourceNotFoundError (-> 404) if the view doesn't exist in the project
                    dataset_view_service.get_dataset_view_by_id(project.id, job_request.dataset_view_id)
                job = ExportDatasetJob(
                    id=job_id,
                    project_id=project.id,
                    params=ExportDatasetJobParams(
                        dataset_id=job_request.dataset_id,
                        dataset_view_id=job_request.dataset_view_id,
                        project_id=project.id,
                        task=project.task,
                        export_format=DatasetFormat(job_request.parameters.export_format),
                        labels=job_request.parameters.filters.labels,
                        subsets=job_request.parameters.filters.subsets,
                        include_unannotated=job_request.parameters.filters.include_unannotated,
                    ),
                )
            case JobType.STAGE_DATASET:
                raise NotImplementedError
            case _:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown job type")
        await job_queue.submit(job)
        return JobView.of(job)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "",
    response_model=list[JobView],
    responses={
        status.HTTP_200_OK: {"description": "List all jobs"},
    },
)
async def list_jobs(job_queue: Annotated[JobQueue, Depends(get_job_queue)]) -> list[JobView]:
    """
    List all the jobs.

    Note that job details are not persisted across server restarts;
    in other words, this endpoint only returns jobs submitted during the current server session.
    """
    return [JobView.of(job) for job in job_queue.list_all()]


@router.get(
    "/{job_id}",
    response_model=JobView,
    responses={
        status.HTTP_200_OK: {"description": "Job found"},
        status.HTTP_404_NOT_FOUND: {"description": "Job with ID not found"},
    },
)
async def get_job(job_id: JobID, job_queue: Annotated[JobQueue, Depends(get_job_queue)]) -> JobView:
    """Get detailed information about a specific job."""
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobView.of(job)


@router.post(
    "/{job_id}:cancel",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobView,
    responses={
        status.HTTP_202_ACCEPTED: {"description": "Job cancel successfully requested"},
        status.HTTP_404_NOT_FOUND: {"description": "Job with ID not found"},
        status.HTTP_409_CONFLICT: {"description": "Job cannot be canceled in its current state"},
    },
)
async def cancel_job(job_id: JobID, job_queue: Annotated[JobQueue, Depends(get_job_queue)]) -> JobView:
    """
    Request cancellation of a specific job.

    This endpoint allows the client to request the cancellation of a job using its unique job ID.
    The cancellation request is processed asynchronously.
    """
    try:
        job, result = job_queue.cancel(job_id)
        match result:
            case CancellationResult.NOT_FOUND:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
            case CancellationResult.IGNORE_CANCEL:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job already completed or cancelled")
            case CancellationResult.PENDING_CANCELLED | CancellationResult.RUNNING_CANCELLING:
                if not job:
                    logger.error("Can't locate job {} after cancellation", job_id)
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Job not found")
                return JobView.of(job)
            case _:
                logger.error("Unexpected cancellation result: {}", result)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error while processing job cancellation"
                )
    except ValueError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Unable to cancel job")


@router.get("/{job_id}/status")
async def stream_job_status(
    job_id: JobID, job_queue: Annotated[JobQueue, Depends(get_job_queue)]
) -> EventSourceResponse:
    """
    Stream real-time status updates for a specific job.

    This endpoint streams job status updates using Server-Sent Events (SSE).
    It sends periodic updates until the job reaches terminal state.
    """
    if not job_queue.get(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return EventSourceResponse(__gen_job_updates(job_id, job_queue))


@router.get("/{job_id}/logs")
async def stream_job_logs(
    job_id: JobID,
    job_dir: Annotated[Path, Depends(get_job_dir)],
    job_queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> EventSourceResponse:
    """
    Stream real-time log output for a specific job.

    This endpoint streams job logs using Server-Sent Events (SSE). It reads
    the job's log file and yields new lines as they are written, allowing clients
    to follow the job's progress in real-time. The stream continues until the
    client disconnects or an error occurs.
    """
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    log_path = job_dir / job.log_file
    if not log_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")

    return EventSourceResponse(__gen_log_stream(job_id, log_path, job_queue))


async def __gen_job_updates(job_id: UUID, job_queue: JobQueue) -> AsyncGenerator[ServerSentEvent]:
    """Generate job status updates."""
    last = None
    while True:
        j = job_queue.get(job_id)
        if not j:
            break
        snap = JobView.of(j).model_dump_json()
        if snap != last:
            yield ServerSentEvent(data=snap)
            last = snap
        if j.status >= JobStatus.DONE:
            break
        await asyncio.sleep(0.1)


async def __gen_log_stream(job_id: UUID, log_path: Path, job_queue: JobQueue) -> AsyncGenerator[ServerSentEvent]:
    """Asynchronously follow a log file and yield new lines as SSE events."""
    try:
        async with aiofiles.open(log_path, encoding="utf-8", errors="replace") as f:
            async for line in f:
                yield ServerSentEvent(data=line.rstrip("\n"))

            while True:
                j = job_queue.get(job_id)
                if not j:
                    break

                line = await f.readline()
                if line:
                    yield ServerSentEvent(data=line.rstrip("\n"))
                    continue

                # No more lines available
                if j.status >= JobStatus.DONE:
                    # Job is done and no new lines - wait briefly to catch any final writes
                    await asyncio.sleep(0.5)
                    final_line = await f.readline()
                    if final_line:
                        yield ServerSentEvent(data=final_line.rstrip("\n"))
                        continue
                    break  # Job is done and no more lines, exit the loop

                # Job still running, wait for more logs
                await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Error reading log file for job")
        yield ServerSentEvent(data="Error reading log file")
