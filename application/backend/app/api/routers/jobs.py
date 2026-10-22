# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
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
from app.api.schemas.jobs.training import TrainingRequest
from app.api.validators import JobID
from app.core.jobs.control_plane import CancellationResult, JobQueue
from app.core.jobs.models import JobStatus
from app.core.logging import extract_log_text
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


def _build_training_job(
    job_id: UUID,
    job_request: TrainingRequest,
    job_dir: Path,
    data_dir: Path,
    project_service: ProjectService,
    system_service: SystemService,
    dataset_view_service: DatasetViewService,
) -> TrainingJob:
    """Resolve the training request parameters into a ready-to-submit training job."""
    device = system_service.training_device(job_request.parameters.device)
    project = project_service.get_project_by_id(job_request.project_id)
    arch_id = job_request.parameters.model_architecture_id
    arch_name = ModelManifestService.get_model_manifest_by_id(arch_id).name
    dataset_view_id = job_request.parameters.dataset_view_id
    dataset_view_name = (
        dataset_view_service.get_dataset_view_by_id(project_id=project.id, dataset_view_id=dataset_view_id).name
        if dataset_view_id is not None
        else None
    )
    return TrainingJob(
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
            dataset_view_id=dataset_view_id,
            dataset_view_name=dataset_view_name,
        ),
    )


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
            "description": "Dataset already in datumaro format and ready for import, "
            "or mutually exclusive parameters were provided"
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
                job = _build_training_job(
                    job_id=job_id,
                    job_request=job_request,
                    job_dir=job_dir,
                    data_dir=data_dir,
                    project_service=project_service,
                    system_service=system_service,
                    dataset_view_service=dataset_view_service,
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


@router.get(
    "/{job_id}/logs",
    response_model=None,
    responses={
        status.HTTP_200_OK: {
            "description": "Job logs, either streamed in real time or downloaded as a file",
            "content": {
                "text/event-stream": {"description": "Real-time stream of NDJSON log records (default)"},
                "text/plain": {"description": "Human-readable log file download (text extracted from NDJSON)"},
                "application/x-ndjson": {"description": "Raw log file download (newline-delimited JSON)"},
            },
        },
        status.HTTP_404_NOT_FOUND: {"description": "Job or log file not found"},
    },
)
async def get_job_logs(
    job_id: JobID,
    job_dir: Annotated[Path, Depends(get_job_dir)],
    job_queue: Annotated[JobQueue, Depends(get_job_queue)],
    accept: Annotated[str | None, Header(description="Accept header to specify the desired response format")] = None,
) -> EventSourceResponse | StreamingResponse:
    """
    Stream or download the log output of a specific job.

    The response format is selected via the Accept header:

    - `text/plain`: download the logs as a human-readable text file, containing only the
      formatted text of each log record.
    - `application/x-ndjson`: download the raw log file, one JSON log record per line.
    - anything else (e.g. `text/event-stream`): stream the logs using Server-Sent Events (SSE).
      The stream reads the job's log file and yields new lines as they are written, allowing
      clients to follow the job's progress in real-time. The stream continues until the job
      finishes, the client disconnects or an error occurs.

    Downloads return the log content available at the time of the request, so they can also
    be used while the job is still running.
    """
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    log_path = job_dir / job.log_file
    if not log_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")

    accept_lower = accept.lower() if accept else ""
    if "text/plain" in accept_lower:
        return StreamingResponse(
            content=__gen_log_file(log_path, as_text=True),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{job.log_file}"'},
        )
    if "application/x-ndjson" in accept_lower:
        # Streamed rather than served via FileResponse: the file may still be growing while the job runs,
        # which would conflict with a Content-Length computed upfront.
        return StreamingResponse(
            content=__gen_log_file(log_path, as_text=False),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{job.log_file}"'},
        )

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


async def __gen_log_file(log_path: Path, as_text: bool) -> AsyncGenerator[str]:
    """Read the current content of a log file, optionally extracting only the human-readable text."""
    async with aiofiles.open(log_path, encoding="utf-8", errors="replace") as f:
        async for line in f:
            if not as_text:
                yield line
            elif text := extract_log_text(line):
                yield text


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
