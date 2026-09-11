# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import time
from uuid import UUID

import requests

from app.api.schemas.jobs import JobView
from app.core.jobs.models import JobStatus, JobType
from app.models import DatasetFormat, TaskType
from tests.bdd.utils.parsers import parse_sse_events

# Overall budget for waiting on a single job to reach a terminal state.
_JOB_WAIT_TIMEOUT_SECONDS = 600
_JOB_RECONNECT_SLEEP_SECONDS = 1


class JobFailedError(Exception):
    """Raised when a job reaches a terminal FAILED state; carries the failed JobView."""

    def __init__(self, job: JobView) -> None:
        self.job = job
        super().__init__(f"Job {job.job_id} failed: {job.error}")


def expect_job_accepted(response: requests.Response) -> JobView:
    assert response.status_code == 202, (
        f"Expected job to be ACCEPTED, but got {response.status_code}, response: {response.text}"
    )
    return JobView.model_validate(response.json())


def wait_for_job_completion(session: requests.Session, base_url: str, job_id: UUID) -> JobView:
    """Wait for a job to reach a terminal state by consuming its SSE status stream.

    If the SSE stream ends before the job reaches a terminal state (e.g. due to
    a transient network/proxy disconnect), the function reconnects and keeps
    waiting, falling back to a regular GET on the job resource so that the test
    does not flake on benign disconnections. A total timeout bounds the wait.
    """
    terminal_statuses = (JobStatus.DONE.name, JobStatus.FAILED.name)
    job: JobView | None = None
    deadline = time.monotonic() + _JOB_WAIT_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        remaining_time = max(0.1, deadline - time.monotonic())

        try:
            read_timeout = min(30.0, remaining_time)
            with session.get(
                f"{base_url}/api/jobs/{job_id}/status",
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=(5, read_timeout),
            ) as stream_response:
                stream_response.raise_for_status()

                for job_data in parse_sse_events(stream_response):
                    job = JobView.model_validate(job_data)
                    if job.status in terminal_statuses:
                        break
        except requests.exceptions.RequestException:
            pass  # Treat as transient; fall through to poll/reconnect.

        if job is not None and job.status in terminal_statuses:
            break

        # SSE stream ended before reaching a terminal state. Re-check the job
        # status directly in case the terminal event was missed, then reconnect.
        remaining_time = max(0.1, deadline - time.monotonic())
        try:
            poll_timeout = min(10.0, remaining_time)
            with requests.get(f"{base_url}/api/jobs/{job_id}", timeout=poll_timeout) as poll_response:
                poll_response.raise_for_status()
                job = JobView.model_validate(poll_response.json())
                if job.status in terminal_statuses:
                    break
        except requests.exceptions.RequestException:
            pass  # Transient failure; will retry on next iteration.

        time.sleep(_JOB_RECONNECT_SLEEP_SECONDS)

    assert job is not None, f"Did not receive any status update for job {job_id}"
    if job.status == JobStatus.FAILED.name:
        raise JobFailedError(job)
    assert job.status == JobStatus.DONE.name, f"Expected job to be DONE, but got {job.status}, error: {job.error}"
    return job


def export_dataset(
    session: requests.Session, base_url: str, project_id: str, export_format: DatasetFormat, filters: str | None = None
) -> JobView:
    job_body = {
        "job_type": JobType.EXPORT_DATASET,
        "project_id": project_id,
        "parameters": {
            "export_format": export_format,
        },
    }
    if filters:
        job_body["parameters"]["filters"] = json.loads(filters)
    response = session.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(session, base_url, job.job_id)


def prepare_dataset(session: requests.Session, base_url: str, staged_dataset_id: str) -> JobView:
    job_body = {
        "job_type": JobType.PREPARE_DATASET_FOR_IMPORT,
        "staged_dataset_id": staged_dataset_id,
    }
    response = session.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(session, base_url, job.job_id)


def import_dataset_to_project(
    session: requests.Session,
    base_url: str,
    project_id: str,
    staged_dataset_id: str,
    labels_mapping: dict[str, str | None] | None = None,
) -> JobView:
    job_body = {
        "job_type": JobType.IMPORT_DATASET_TO_PROJECT,
        "project_id": project_id,
        "staged_dataset_id": staged_dataset_id,
        "parameters": {},
    }
    if labels_mapping is not None:
        job_body["parameters"] = {"labels_mapping": labels_mapping}
    response = session.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(session, base_url, job.job_id)


def import_dataset_as_new_project(
    session: requests.Session,
    base_url: str,
    project_name: str,
    staged_dataset_id: str,
    labels: list[str],
    task_type: TaskType,
    exclusive_labels: bool = False,
    include_unannotated: bool = True,
) -> JobView:
    job_body = {
        "job_type": JobType.IMPORT_DATASET_AS_NEW_PROJECT,
        "staged_dataset_id": staged_dataset_id,
        "parameters": {
            "project": {
                "name": project_name,
                "task_type": task_type,
                "exclusive_labels": exclusive_labels,
            },
            "filters": {
                "labels": labels,
                "include_unannotated": include_unannotated,
            },
        },
    }
    response = session.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(session, base_url, job.job_id)


def train(session: requests.Session, base_url: str, project_id: UUID, model_id: str, device: str) -> JobView:
    """Launch a training job for *model_id* on *device* and wait for completion."""
    job_body = {
        "job_type": JobType.TRAIN,
        "project_id": str(project_id),
        "parameters": {
            "device": device,
            "model_architecture_id": model_id,
        },
    }
    response = session.post(f"{base_url}/api/jobs", json=job_body)
    job = expect_job_accepted(response)
    return wait_for_job_completion(session, base_url, job.job_id)
