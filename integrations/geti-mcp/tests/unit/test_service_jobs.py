# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for job monitoring, training submission, and cancellation."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from geti_mcp.config import Limits, Permissions
from geti_mcp.errors import ErrorCode, GetiMcpError
from tests.conftest import (
    ARCHITECTURE,
    JOB_A,
    JOB_NO_OWNER,
    JOB_OTHER_PROJECT,
    MODEL_A,
    PROJECT_A,
    PROJECT_B,
    REVISION_A,
    FakeGeti,
    make_config,
)

TRAINING_ALLOWED = Permissions(allow_training=True)
CANCEL_ALLOWED = Permissions(allow_job_cancellation=True)
FAST_WAIT = Limits(max_wait_seconds=1.0)


def _set_status(fake: FakeGeti, job_id: str, status: str, **extra) -> None:
    job = dict(fake.jobs[job_id])
    job["status"] = status
    job.update(extra)
    fake.jobs[job_id] = job


class TestListJobs:
    async def test_jobs_outside_the_scope_are_excluded(self, service) -> None:
        jobs = await service.list_jobs()
        assert [job.job_id for job in jobs.jobs] == [JOB_A]
        assert jobs.excluded_unauthorized == 2

    async def test_jobs_with_unknown_ownership_are_excluded_not_guessed(self, make_service) -> None:
        """Geti lists all jobs globally, so a job whose owner cannot be resolved is dropped."""
        jobs = await make_service(make_config(allowed_projects=None)).list_jobs()
        assert JOB_NO_OWNER not in [job.job_id for job in jobs.jobs]
        assert jobs.excluded_unauthorized == 1

    async def test_project_filter_must_itself_be_authorized(self, service, fake_geti: FakeGeti) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.list_jobs(project_id=PROJECT_B)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert fake_geti.requests == []

    async def test_job_type_filter_is_applied(self, service) -> None:
        assert (await service.list_jobs(job_type="quantize")).jobs == []
        assert (await service.list_jobs(job_type="train")).jobs

    async def test_invalid_job_type_is_rejected_with_the_valid_options(self, service) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.list_jobs(job_type="training")
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "train" in excinfo.value.message

    async def test_listing_is_bounded(self, fake_geti: FakeGeti, make_service) -> None:
        for index in range(60):
            job_id = f"{index:08d}-1111-2222-3333-444444444444"
            fake_geti.jobs[job_id] = {
                "job_id": job_id,
                "job_type": "train",
                "status": "DONE",
                "progress": 100.0,
                "metadata": {"project": {"id": PROJECT_A}},
            }
        jobs = await make_service(make_config(limits=Limits(max_items=10))).list_jobs()
        assert jobs.returned == 10
        assert jobs.truncated is True


class TestGetJob:
    async def test_reports_backend_status_and_normalized_state(self, service) -> None:
        job = await service.get_job(JOB_A)
        assert job.backend_status == "RUNNING"
        assert job.state == "running"
        assert job.terminal is False
        assert job.progress == 42.0

    async def test_exposes_training_context(self, service) -> None:
        job = await service.get_job(JOB_A)
        assert job.model_architecture_id == ARCHITECTURE
        assert job.dataset_revision_id == REVISION_A
        assert job.device == "cpu"

    async def test_job_in_another_project_is_denied(self, service) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.get_job(JOB_OTHER_PROJECT)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED

    async def test_job_with_unknown_owner_fails_closed(self, make_service) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await make_service(make_config(allowed_projects=None)).get_job(JOB_NO_OWNER)
        assert excinfo.value.code is ErrorCode.OWNERSHIP_UNKNOWN

    async def test_missing_job_explains_that_records_may_not_survive_a_restart(self, service) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.get_job("99999999-0000-0000-0000-000000000000")
        assert excinfo.value.code is ErrorCode.NOT_FOUND
        assert "restart" in (excinfo.value.guidance or "").lower()

    async def test_failed_job_reports_the_failure_without_leaking_paths(
        self, fake_geti: FakeGeti, make_service
    ) -> None:
        _set_status(
            fake_geti,
            JOB_A,
            "FAILED",
            error="CUDA OOM while writing /Users/alice/geti/data/projects/x/model.pth",
        )
        job = await make_service().get_job(JOB_A)
        assert job.state == "failed"
        assert job.terminal is True
        assert job.error is not None
        assert "alice" not in job.error

    async def test_successful_job_is_terminal(self, fake_geti: FakeGeti, make_service) -> None:
        _set_status(fake_geti, JOB_A, "DONE", progress=100.0, finished_at="2026-01-02T10:40:00Z")
        job = await make_service().get_job(JOB_A)
        assert (job.state, job.terminal) == ("succeeded", True)

    async def test_unknown_backend_status_is_not_guessed(self, fake_geti: FakeGeti, make_service) -> None:
        _set_status(fake_geti, JOB_A, "SOMETHING_NEW")
        job = await make_service().get_job(JOB_A)
        assert job.backend_status == "SOMETHING_NEW"
        assert job.state == "unknown"
        assert job.terminal is False


class TestWaitForJob:
    async def test_returns_immediately_when_already_terminal(self, fake_geti: FakeGeti, make_service) -> None:
        _set_status(fake_geti, JOB_A, "DONE")
        result = await make_service().wait_for_job(JOB_A, 10.0)
        assert result.reached_terminal is True
        assert result.timed_out is False
        assert result.polls == 1

    async def test_timeout_reports_the_job_is_still_running(self, make_service) -> None:
        result = await make_service(make_config(limits=FAST_WAIT)).wait_for_job(JOB_A, 1.0)
        assert result.timed_out is True
        assert result.reached_terminal is False
        assert result.job.state == "running"
        assert "continues running" in result.summary.lower()

    async def test_timeout_never_cancels_the_job(self, fake_geti: FakeGeti, make_service) -> None:
        await make_service(make_config(limits=FAST_WAIT)).wait_for_job(JOB_A, 1.0)
        assert not [r for r in fake_geti.requests if r.method == "POST"]
        assert fake_geti.jobs[JOB_A]["status"] == "RUNNING"

    async def test_observes_a_transition_to_terminal(self, fake_geti: FakeGeti, make_service) -> None:
        polls = {"count": 0}

        def progressing(request: httpx.Request) -> httpx.Response:
            polls["count"] += 1
            status = "DONE" if polls["count"] >= 2 else "RUNNING"
            job = dict(fake_geti.jobs[JOB_A])
            job["status"] = status
            return httpx.Response(200, json=job)

        fake_geti.override(rf"/api/jobs/{JOB_A}$", progressing)
        result = await make_service(make_config(limits=FAST_WAIT)).wait_for_job(JOB_A, 1.0)
        assert result.reached_terminal is True
        assert result.job.state == "succeeded"
        assert result.polls >= 2

    async def test_cancelling_is_not_treated_as_terminal(self, fake_geti: FakeGeti, make_service) -> None:
        _set_status(fake_geti, JOB_A, "CANCELLING")
        result = await make_service(make_config(limits=FAST_WAIT)).wait_for_job(JOB_A, 1.0)
        assert result.job.state == "cancelling"
        assert result.reached_terminal is False
        assert result.timed_out is True

    async def test_terminal_cancellation_ends_the_wait(self, fake_geti: FakeGeti, make_service) -> None:
        _set_status(fake_geti, JOB_A, "CANCELLED")
        result = await make_service(make_config(limits=FAST_WAIT)).wait_for_job(JOB_A, 1.0)
        assert result.job.state == "cancelled"
        assert result.reached_terminal is True

    async def test_wait_is_clamped_to_the_configured_maximum(self, make_service) -> None:
        result = await make_service(make_config(limits=FAST_WAIT)).wait_for_job(JOB_A, 10_000.0)
        assert result.waited_seconds <= 1.5

    async def test_a_disconnected_client_leaves_the_job_untouched(self, fake_geti: FakeGeti, make_service) -> None:
        """Abandoning the wait must never be interpreted as a cancellation request."""
        service = make_service(make_config(limits=Limits(max_wait_seconds=30.0)))
        task = asyncio.create_task(service.wait_for_job(JOB_A, 30.0))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not [r for r in fake_geti.requests if r.method == "POST"]
        assert fake_geti.jobs[JOB_A]["status"] == "RUNNING"

    async def test_unauthorized_job_is_denied(self, service) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.wait_for_job(JOB_OTHER_PROJECT, 1.0)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED


class TestStartTraining:
    def _service(self, make_service, **kwargs):
        return make_service(make_config(permissions=TRAINING_ALLOWED, **kwargs))

    async def test_submission_requires_the_opt_in(self, service, fake_geti: FakeGeti) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.start_training(PROJECT_A, ARCHITECTURE, "cpu")
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert fake_geti.requests == []

    async def test_submission_is_denied_for_unauthorized_projects(self, make_service, fake_geti: FakeGeti) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await self._service(make_service).start_training(PROJECT_B, ARCHITECTURE, "cpu")
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert fake_geti.requests == []

    async def test_accepted_submission_returns_a_job_id(self, make_service, fake_geti: FakeGeti) -> None:
        result = await self._service(make_service).start_training(PROJECT_A, ARCHITECTURE, "cpu")
        assert result.job_id == JOB_A
        assert result.state == "queued"
        assert fake_geti.submitted_training == [
            {
                "job_type": "train",
                "project_id": PROJECT_A,
                "parameters": {"device": "cpu", "model_architecture_id": ARCHITECTURE},
            }
        ]

    async def test_unset_optional_parameters_are_omitted_not_nulled(self, make_service, fake_geti: FakeGeti) -> None:
        """Omitting them lets Geti apply its own defaults rather than this server choosing."""
        await self._service(make_service).start_training(PROJECT_A, ARCHITECTURE, "cpu")
        parameters = fake_geti.submitted_training[0]["parameters"]
        assert "parent_model_revision_id" not in parameters
        assert "dataset_revision_id" not in parameters

    async def test_optional_parameters_are_forwarded(self, make_service, fake_geti: FakeGeti) -> None:
        await self._service(make_service).start_training(
            PROJECT_A,
            ARCHITECTURE,
            "xpu-0",
            parent_model_revision_id=MODEL_A,
            dataset_revision_id=REVISION_A,
        )
        parameters = fake_geti.submitted_training[0]["parameters"]
        assert parameters["parent_model_revision_id"] == MODEL_A
        assert parameters["dataset_revision_id"] == REVISION_A
        assert parameters["device"] == "xpu-0"

    async def test_preflight_facts_are_reported_as_non_atomic(self, make_service) -> None:
        result = await self._service(make_service).start_training(PROJECT_A, ARCHITECTURE, "cpu")
        assert result.preflight
        assert any("atomic" in note.lower() for note in result.preflight)

    async def test_unknown_architecture_is_refused_without_substituting_one(
        self, make_service, fake_geti: FakeGeti
    ) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await self._service(make_service).start_training(PROJECT_A, "made-up-architecture", "cpu")
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert fake_geti.submitted_training == []

    async def test_unavailable_device_is_refused_without_falling_back_to_cpu(
        self, make_service, fake_geti: FakeGeti
    ) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await self._service(make_service).start_training(PROJECT_A, ARCHITECTURE, "cuda-7")
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "cpu" in str(excinfo.value.details)
        assert fake_geti.submitted_training == []

    async def test_untested_backend_version_blocks_submission(self, make_service, fake_geti: FakeGeti) -> None:
        fake_geti.api_version = "9.9.9"
        with pytest.raises(GetiMcpError) as excinfo:
            await self._service(make_service).start_training(PROJECT_A, ARCHITECTURE, "cpu")
        assert excinfo.value.code is ErrorCode.INCOMPATIBLE_BACKEND
        assert fake_geti.submitted_training == []

    async def test_backend_rejection_is_surfaced_verbatim(self, make_service, fake_geti: FakeGeti) -> None:
        fake_geti.override(
            r"/api/jobs$",
            lambda request: httpx.Response(409, json={"detail": "A training job is already running."})
            if request.method == "POST"
            else httpx.Response(200, json=[]),
        )
        with pytest.raises(GetiMcpError) as excinfo:
            await self._service(make_service).start_training(PROJECT_A, ARCHITECTURE, "cpu")
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "already running" in str(excinfo.value.details)

    async def test_interrupted_submission_is_ambiguous_and_not_retried(self, make_service, fake_geti: FakeGeti) -> None:
        attempts = {"count": 0}

        def timeout(request: httpx.Request) -> httpx.Response:
            if request.method != "POST":
                return httpx.Response(200, json=list(fake_geti.jobs.values()))
            attempts["count"] += 1
            raise httpx.ReadTimeout("timed out", request=request)

        fake_geti.override(r"/api/jobs$", timeout)
        with pytest.raises(GetiMcpError) as excinfo:
            await self._service(make_service).start_training(PROJECT_A, ARCHITECTURE, "cpu")
        assert excinfo.value.code is ErrorCode.SUBMISSION_OUTCOME_UNKNOWN
        assert attempts["count"] == 1
        assert "do not resubmit" in (excinfo.value.guidance or "").lower()


class TestCancelJob:
    def _service(self, make_service):
        return make_service(make_config(permissions=CANCEL_ALLOWED))

    async def test_cancellation_requires_the_opt_in(self, service, fake_geti: FakeGeti) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.cancel_job(JOB_A)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert fake_geti.requests == []

    async def test_ownership_is_checked_before_mutating(self, make_service, fake_geti: FakeGeti) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await self._service(make_service).cancel_job(JOB_OTHER_PROJECT)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert not [r for r in fake_geti.requests if r.method == "POST"]

    async def test_unknown_ownership_blocks_cancellation(self, make_service, fake_geti: FakeGeti) -> None:
        service = make_service(make_config(allowed_projects=None, permissions=CANCEL_ALLOWED))
        with pytest.raises(GetiMcpError) as excinfo:
            await service.cancel_job(JOB_NO_OWNER)
        assert excinfo.value.code is ErrorCode.OWNERSHIP_UNKNOWN
        assert not [r for r in fake_geti.requests if r.method == "POST"]

    async def test_request_accepted_is_not_reported_as_terminated(self, make_service) -> None:
        result = await self._service(make_service).cancel_job(JOB_A)
        assert result.cancellation_requested is True
        assert result.backend_status == "CANCELLING"
        assert result.state == "cancelling"
        assert result.terminated is False
        assert "confirm" in result.summary.lower() or "poll" in result.summary.lower()

    async def test_already_terminal_job_is_not_mutated(self, fake_geti: FakeGeti, make_service) -> None:
        _set_status(fake_geti, JOB_A, "CANCELLED")
        result = await self._service(make_service).cancel_job(JOB_A)
        assert result.terminated is True
        assert result.cancellation_requested is False
        assert not [r for r in fake_geti.requests if r.method == "POST"]
