# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from app.models.jobs.quantization_job import QuantizationJob, QuantizationJobParams


@pytest.fixture
def fxt_quantization_params() -> Callable[[UUID, UUID], QuantizationJobParams]:
    def _make_quantization_job_params(job_id: UUID, project_id: UUID) -> QuantizationJobParams:
        return QuantizationJobParams(
            job_id=job_id,
            project_id=project_id,
            model_id=uuid4(),
            model_architecture_id="test_arch",
            model_architecture_name="Test Arch",
        )

    return _make_quantization_job_params


@pytest.fixture
def fxt_quantization_job(tmp_path, fxt_quantization_params):
    job_id = uuid4()
    project_id = uuid4()
    log_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"
    log_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    return QuantizationJob(
        id=job_id,
        project_id=project_id,
        log_dir=log_dir,
        data_dir=data_dir,
        params=fxt_quantization_params(job_id, project_id),
    )


class TestQuantizationJob:
    def test_on_complete_copies_log_file(self, fxt_quantization_job):
        """Test that log file is copied to the variant's directory."""
        log_path = fxt_quantization_job.log_dir / fxt_quantization_job.log_file
        log_path.write_text("Quantization log content")

        fxt_quantization_job.on_complete()

        expected_path = (
            fxt_quantization_job.data_dir
            / "projects"
            / str(fxt_quantization_job.project_id)
            / "models"
            / str(fxt_quantization_job.params.model_id)
            / "variants"
            / str(fxt_quantization_job.params.model_variant_id)
            / "quantization.log"
        )
        assert expected_path.exists()
        assert expected_path.read_text() == "Quantization log content"

    @patch("app.models.jobs.quantization_job.logger")
    def test_on_complete_logs_warning(self, mock_logger, fxt_quantization_job):
        """A warning is logged and no file copied when the source log file doesn't exist."""
        fxt_quantization_job.on_complete()

        log_path = fxt_quantization_job.log_dir / fxt_quantization_job.log_file
        mock_logger.warning.assert_called_once_with("Log file {} does not exist", log_path)
        expected_path = (
            fxt_quantization_job.data_dir
            / "projects"
            / str(fxt_quantization_job.project_id)
            / "models"
            / str(fxt_quantization_job.params.model_id)
            / "variants"
            / str(fxt_quantization_job.params.model_variant_id)
            / "quantization.log"
        )
        assert not expected_path.exists()

    def test_on_complete_removes_getitune_workspace(self, fxt_quantization_job):
        """Test that the getitune quantization workspace directory is removed on job completion."""
        workspace_dir = (
            fxt_quantization_job.data_dir / f"getitune-quantize-workspace-{fxt_quantization_job.params.model_id}"
        )
        timestamp_dir = workspace_dir / "20260101_000000"
        timestamp_dir.mkdir(parents=True)
        (timestamp_dir / "leftover.txt").write_text("temp")

        fxt_quantization_job.on_complete()

        assert not workspace_dir.exists()

    def test_on_complete_no_op_when_workspace_missing(self, fxt_quantization_job):
        """on_complete must not raise when the getitune workspace was never created."""
        workspace_dir = (
            fxt_quantization_job.data_dir / f"getitune-quantize-workspace-{fxt_quantization_job.params.model_id}"
        )
        assert not workspace_dir.exists()

        # Should not raise
        fxt_quantization_job.on_complete()
