# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from starlette import status

from app.api.dependencies import get_data_dir, get_job_dir, get_job_queue
from app.api.schemas.jobs import JobRequestAdapter
from app.api.schemas.jobs.quantization import QuantizationRequest
from app.api.schemas.jobs.training import TrainingRequest
from app.core.jobs import JobQueue
from app.core.jobs.control_plane import CancellationResult
from app.core.jobs.models import Job, JobStatus, JobType
from app.models import Project, Task, TaskType, TrainingJob, TrainingJobParams
from app.models.system import DeviceInfo, DeviceType
from app.services.base import ResourceNotFoundError, ResourceType


async def stream_test(client: AsyncClient, job_id: UUID, path: str = "logs") -> list[str]:
    events = []
    async with client.stream("GET", f"/api/jobs/{job_id}/{path}") as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(line)
    return events


@pytest.fixture
def fxt_jobs_queue(fxt_app) -> Mock:
    jobs_queue = Mock(spec=JobQueue)
    fxt_app.dependency_overrides[get_job_queue] = lambda: jobs_queue
    return jobs_queue


@pytest.fixture
def fxt_job() -> Callable[[UUID | None, JobStatus, float], Job]:
    def job_factory(
        job_id: UUID | None = None, job_status: JobStatus = JobStatus.RUNNING, progress: float = 50.0
    ) -> Job:
        job_id_ = job_id or uuid4()
        project_id_ = uuid4()
        return TrainingJob(
            id=job_id_,
            project_id=project_id_,
            log_dir=Path(""),
            data_dir=Path(""),
            status=job_status,
            progress=100.0 if job_status >= JobStatus.DONE else progress,
            job_type=JobType.TRAIN,
            params=TrainingJobParams(
                device=DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None),
                job_id=job_id_,
                project_id=project_id_,
                model_architecture_id="test_arch",
                model_architecture_name="TestArch",
                task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
            ),
        )

    return job_factory


class TestJobEndpoints:
    def test_submit_train_job(self, fxt_app, tmp_path, fxt_client, fxt_jobs_queue, fxt_project_service):
        fxt_app.dependency_overrides[get_job_dir] = lambda: tmp_path / "logs" / "jobs"
        fxt_app.dependency_overrides[get_data_dir] = lambda: tmp_path / "data"
        project = Mock(spec=Project)
        project.id = uuid4()
        project.task = Mock(spec=Task)
        project.task.task_type = TaskType.CLASSIFICATION
        project.task.exclusive_labels = True
        fxt_project_service.get_project_by_id.return_value = project

        mock_manifest = Mock()
        mock_manifest.name = "ViT Tiny"

        with patch("app.api.routers.jobs.ModelManifestService.get_model_manifest_by_id", return_value=mock_manifest):
            job_request = JobRequestAdapter.validate_python(
                {
                    "project_id": project.id,
                    "job_type": JobType.TRAIN,
                    "parameters": {
                        "device": "cpu",
                        "model_architecture_id": "image-classification-vit-tiny",
                        "parent_model_revision_id": uuid4(),
                        "parent_model_variant_id": uuid4(),
                    },
                }
            )

            response = fxt_client.post("/api/jobs", json=job_request.model_dump(mode="json"))

            assert response.status_code == status.HTTP_202_ACCEPTED
            response_json = response.json()
            assert response_json["job_id"]
            assert response_json["metadata"]["device"]["name"] == "CPU"
            assert re.match(r"^ViT Tiny \([0-9a-f]{8}\)$", response_json["metadata"]["model"]["name"])
            job_request = cast(TrainingRequest, job_request)
            fxt_project_service.get_project_by_id.assert_called_once_with(job_request.project_id)
            fxt_jobs_queue.submit.assert_called_once()
            assert fxt_jobs_queue.submit.call_args[0][0].params.model_architecture_id == "image-classification-vit-tiny"
            assert fxt_jobs_queue.submit.call_args[0][0].params.task.task_type == TaskType.CLASSIFICATION

    def test_submit_quantize_job(self, fxt_app, tmp_path, fxt_client, fxt_jobs_queue, fxt_project_service):
        fxt_app.dependency_overrides[get_job_dir] = lambda: tmp_path / "logs" / "jobs"
        fxt_app.dependency_overrides[get_data_dir] = lambda: tmp_path / "data"
        project = Mock(spec=Project)
        project.id = uuid4()
        project.task = Mock(spec=Task)
        project.task.task_type = TaskType.CLASSIFICATION
        project.task.exclusive_labels = True
        fxt_project_service.get_project_by_id.return_value = project

        model_id = uuid4()
        mock_manifest = Mock()
        mock_manifest.name = "ViT Tiny"
        job_request = JobRequestAdapter.validate_python(
            {
                "project_id": project.id,
                "job_type": JobType.QUANTIZE,
                "parameters": {
                    "model_id": model_id,
                    "model_architecture_id": "image-classification-vit-tiny",
                    "max_calibration_subset_size": 200,
                    "max_drop": 0.01,
                    "max_num_iterations": 5,
                },
            }
        )

        with patch("app.api.routers.jobs.ModelManifestService.get_model_manifest_by_id", return_value=mock_manifest):
            response = fxt_client.post("/api/jobs", json=job_request.model_dump(mode="json"))

        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json()["job_id"]
        job_request = cast(QuantizationRequest, job_request)
        fxt_project_service.get_project_by_id.assert_called_once_with(job_request.project_id)
        fxt_jobs_queue.submit.assert_called_once()
        submitted_job = fxt_jobs_queue.submit.call_args[0][0]
        assert submitted_job.job_type == JobType.QUANTIZE
        assert submitted_job.params.model_id == model_id
        assert submitted_job.params.model_architecture_id == "image-classification-vit-tiny"
        assert submitted_job.params.model_architecture_name == "ViT Tiny"
        assert submitted_job.params.max_calibration_subset_size == 200
        assert submitted_job.params.max_drop == 0.01
        assert submitted_job.params.max_num_iterations == 5
        assert submitted_job.project_id == project.id

    def test_submit_quantize_job_defaults(self, fxt_app, tmp_path, fxt_client, fxt_jobs_queue, fxt_project_service):
        """Quantize request without max_drop uses defaults (None for max_drop, 100 for subset size)."""
        fxt_app.dependency_overrides[get_job_dir] = lambda: tmp_path / "logs" / "jobs"
        fxt_app.dependency_overrides[get_data_dir] = lambda: tmp_path / "data"
        project = Mock(spec=Project)
        project.id = uuid4()
        project.task = Mock(spec=Task)
        project.task.task_type = TaskType.DETECTION
        project.task.exclusive_labels = True
        fxt_project_service.get_project_by_id.return_value = project

        model_id = uuid4()
        mock_manifest = Mock()
        mock_manifest.name = "ViT Tiny"
        job_request = JobRequestAdapter.validate_python(
            {
                "project_id": project.id,
                "job_type": JobType.QUANTIZE,
                "parameters": {
                    "model_id": model_id,
                    "model_architecture_id": "image-classification-vit-tiny",
                },
            }
        )

        with patch("app.api.routers.jobs.ModelManifestService.get_model_manifest_by_id", return_value=mock_manifest):
            response = fxt_client.post("/api/jobs", json=job_request.model_dump(mode="json"))

        assert response.status_code == status.HTTP_202_ACCEPTED
        submitted_job = fxt_jobs_queue.submit.call_args[0][0]
        assert submitted_job.job_type == JobType.QUANTIZE
        assert submitted_job.params.model_id == model_id
        assert submitted_job.params.max_calibration_subset_size == 100
        assert submitted_job.params.max_drop is None
        assert submitted_job.params.max_num_iterations == 10

    def test_submit_export_dataset_job_with_dataset_view(
        self, fxt_app, tmp_path, fxt_client, fxt_jobs_queue, fxt_project_service, fxt_dataset_view_service
    ):
        fxt_app.dependency_overrides[get_job_dir] = lambda: tmp_path / "logs" / "jobs"
        fxt_app.dependency_overrides[get_data_dir] = lambda: tmp_path / "data"
        project = Mock(spec=Project)
        project.id = uuid4()
        project.task = Mock(spec=Task)
        project.task.task_type = TaskType.DETECTION
        project.task.exclusive_labels = True
        fxt_project_service.get_project_by_id.return_value = project
        dataset_view_id = uuid4()
        fxt_dataset_view_service.get_dataset_view_by_id.return_value = Mock()

        response = fxt_client.post(
            "/api/jobs",
            json={
                "project_id": str(project.id),
                "job_type": "export_dataset",
                "dataset_view_id": str(dataset_view_id),
                "parameters": {"export_format": "coco", "filters": {}},
            },
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        fxt_dataset_view_service.get_dataset_view_by_id.assert_called_once_with(project.id, dataset_view_id)
        submitted_job = fxt_jobs_queue.submit.call_args[0][0]
        assert submitted_job.params.dataset_view_id == dataset_view_id
        assert submitted_job.params.dataset_id is None

    def test_submit_export_dataset_job_with_unknown_dataset_view(
        self, fxt_app, tmp_path, fxt_client, fxt_jobs_queue, fxt_project_service, fxt_dataset_view_service
    ):
        fxt_app.dependency_overrides[get_job_dir] = lambda: tmp_path / "logs" / "jobs"
        fxt_app.dependency_overrides[get_data_dir] = lambda: tmp_path / "data"
        project = Mock(spec=Project)
        project.id = uuid4()
        project.task = Mock(spec=Task)
        project.task.task_type = TaskType.DETECTION
        project.task.exclusive_labels = True
        fxt_project_service.get_project_by_id.return_value = project
        dataset_view_id = uuid4()
        fxt_dataset_view_service.get_dataset_view_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_VIEW, str(dataset_view_id)
        )

        response = fxt_client.post(
            "/api/jobs",
            json={
                "project_id": str(project.id),
                "job_type": "export_dataset",
                "dataset_view_id": str(dataset_view_id),
                "parameters": {"export_format": "coco", "filters": {}},
            },
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_jobs_queue.submit.assert_not_called()

    def test_submit_export_dataset_job_with_dataset_id_and_dataset_view_id(
        self, fxt_app, tmp_path, fxt_client, fxt_jobs_queue, fxt_project_service, fxt_dataset_view_service
    ):
        """dataset_id and dataset_view_id are mutually exclusive."""
        fxt_app.dependency_overrides[get_job_dir] = lambda: tmp_path / "logs" / "jobs"
        fxt_app.dependency_overrides[get_data_dir] = lambda: tmp_path / "data"
        project = Mock(spec=Project)
        project.id = uuid4()
        fxt_project_service.get_project_by_id.return_value = project

        response = fxt_client.post(
            "/api/jobs",
            json={
                "project_id": str(project.id),
                "job_type": "export_dataset",
                "dataset_id": str(uuid4()),
                "dataset_view_id": str(uuid4()),
                "parameters": {"export_format": "coco", "filters": {}},
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_jobs_queue.submit.assert_not_called()

    def test_list_jobs(self, fxt_client, fxt_jobs_queue, fxt_job):
        fxt_jobs_queue.list_all.return_value = [fxt_job(), fxt_job()]

        response = fxt_client.get("/api/jobs")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_jobs_queue.list_all.assert_called_once()

    def test_get_job(self, fxt_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        fxt_jobs_queue.get.return_value = fxt_job(job_id)

        response = fxt_client.get(f"/api/jobs/{job_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["job_id"] == str(job_id)
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    def test_get_job_not_found(self, fxt_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        fxt_jobs_queue.get.return_value = None

        response = fxt_client.get(f"/api/jobs/{job_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    @pytest.mark.parametrize(
        "job_id, cancellation_result, expected_status",
        [
            (uuid4(), CancellationResult.PENDING_CANCELLED, status.HTTP_202_ACCEPTED),
            (uuid4(), CancellationResult.RUNNING_CANCELLING, status.HTTP_202_ACCEPTED),
            (uuid4(), CancellationResult.IGNORE_CANCEL, status.HTTP_409_CONFLICT),
            (uuid4(), CancellationResult.NOT_FOUND, status.HTTP_404_NOT_FOUND),
        ],
    )
    def test_cancel_job(self, job_id, cancellation_result, expected_status, fxt_client, fxt_jobs_queue, fxt_job):
        fxt_jobs_queue.cancel.return_value = fxt_job(job_id), cancellation_result

        response = fxt_client.post(f"/api/jobs/{job_id}:cancel")

        assert response.status_code == expected_status
        if expected_status == status.HTTP_202_ACCEPTED:
            assert response.json()["job_id"] == str(job_id)
        fxt_jobs_queue.cancel.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_stream_job_status_not_found(self, fxt_async_client, fxt_jobs_queue):
        job_id = uuid4()
        fxt_jobs_queue.get.return_value = None

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/status")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_stream_job_status_stops_when_done(self, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        # Simulate a job that completes after a few updates
        job_running = fxt_job(job_id, JobStatus.RUNNING, progress=50.0)
        job_done = fxt_job(job_id, JobStatus.DONE, progress=100.0)

        # First call returns running job, subsequent calls return done job
        fxt_jobs_queue.get.side_effect = [job_running, job_running, job_done, None]

        events = await asyncio.wait_for(stream_test(fxt_async_client, job_id, "status"), 3)
        assert len(events) == 2
        # Verify the stream contains job status updates
        assert '"status":"RUNNING"' in events[0]
        assert '"status":"DONE"' in events[1]

    @pytest.mark.asyncio
    async def test_stream_job_status_yields_only_changed_updates(self, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_v1 = fxt_job(job_id, JobStatus.RUNNING, progress=25.0)
        job_v2 = fxt_job(job_id, JobStatus.RUNNING, progress=75.0)
        job_done = fxt_job(job_id, JobStatus.DONE, progress=100.0)

        # Return same job twice (no change), then changed job, then done
        fxt_jobs_queue.get.side_effect = [job_v1, job_v1, job_v2, job_done, None]

        events = await asyncio.wait_for(stream_test(fxt_async_client, job_id, "status"), 3)
        # Should get at least 2 events (initial and one change)
        assert len(events) == 3
        assert '"progress":25.0' in events[0]
        assert '"progress":75.0' in events[1]
        assert '"progress":100' in events[2]

    @pytest.mark.asyncio
    async def test_stream_job_logs_yields_log_lines(self, fxt_app, tmp_path, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        job_v1 = fxt_job(job_id, JobStatus.RUNNING, progress=25.0)
        job_v2 = fxt_job(job_id, JobStatus.RUNNING, progress=75.0)
        job_done = fxt_job(job_id, JobStatus.DONE, progress=100.0)
        # Create a log file with some content using the job's log_file property
        log_file = job_dir / job_v1.log_file
        log_file.write_text("Line 1\nLine 2\nLine 3\n")
        fxt_jobs_queue.get.side_effect = [job_v1, job_v1, job_v2, job_done, None]

        fxt_app.dependency_overrides[get_job_dir] = lambda: job_dir

        events = await asyncio.wait_for(stream_test(fxt_async_client, job_id), 2)
        assert len(events) == 3
        assert "Line 1" in events[0]
        assert "Line 2" in events[1]
        assert "Line 3" in events[2]

    @pytest.mark.asyncio
    async def test_stream_job_logs_not_found(self, fxt_app, tmp_path, fxt_async_client, fxt_jobs_queue):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        fxt_jobs_queue.get.return_value = None

        fxt_app.dependency_overrides[get_job_dir] = lambda: job_dir

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/logs")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_jobs_queue.get.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_stream_job_logs_completed(self, fxt_app, tmp_path, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        job = fxt_job(job_id, JobStatus.DONE)
        fxt_jobs_queue.get.return_value = job

        log_file = job_dir / job.log_file
        log_file.write_text("Line 1\nLine 2\nLine 3\n")

        fxt_app.dependency_overrides[get_job_dir] = lambda: job_dir

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/logs")

        assert response.status_code == status.HTTP_200_OK

        events = await asyncio.wait_for(stream_test(fxt_async_client, job_id), 1)
        assert len(events) == 3
        assert "Line 1" in events[0]
        assert "Line 2" in events[1]
        assert "Line 3" in events[2]

    @pytest.mark.asyncio
    async def test_stream_job_logs_file_not_found(self, fxt_app, tmp_path, fxt_async_client, fxt_jobs_queue, fxt_job):
        job_id = uuid4()
        job_dir = tmp_path / "logs" / "jobs"
        job_dir.mkdir(parents=True)

        job = fxt_job(job_id, JobStatus.RUNNING)
        # Don't create the log file - it should not exist
        fxt_jobs_queue.get.return_value = job

        fxt_app.dependency_overrides[get_job_dir] = lambda: job_dir

        response = await fxt_async_client.get(f"/api/jobs/{job_id}/logs")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "log file not found" in response.json()["detail"].lower()
