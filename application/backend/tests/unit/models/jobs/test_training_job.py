# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest

from app.models import Task, TaskType, TrainingJob, TrainingJobParams, TrainingStatus
from app.models.system import DeviceInfo, DeviceType
from app.services import ResourceNotFoundError, ResourceType


@pytest.fixture
def fxt_training_params() -> Callable[[UUID, UUID], TrainingJobParams]:
    def _make_training_job_params(job_id: UUID, project_id: UUID) -> TrainingJobParams:
        return TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            model_architecture_id="test_arch",
            model_architecture_name="Test Arch",
            task=Task(task_type=TaskType.CLASSIFICATION),
            job_id=job_id,
            project_id=project_id,
        )

    return _make_training_job_params


@pytest.fixture
def fxt_training_job(tmp_path, fxt_training_params):
    job_id = uuid4()
    project_id = uuid4()
    log_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"
    log_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    return TrainingJob(
        id=job_id,
        project_id=project_id,
        log_dir=log_dir,
        data_dir=data_dir,
        params=fxt_training_params(job_id, project_id),
    )


class TestTrainingJob:
    @pytest.fixture(autouse=True)
    def fxt_skip_status_reconciliation(self):
        """These tests only cover log/workspace cleanup; the reconciliation step has its own test class."""
        with patch.object(TrainingJob, "_reconcile_stuck_training_status"):
            yield

    def test_on_complete_copies_log_file(self, fxt_training_job):
        """Test that log file is copied to the correct destination."""
        # Create the log file
        log_path = fxt_training_job.log_dir / fxt_training_job.log_file
        log_path.write_text("Training log content")

        # Execute
        fxt_training_job.on_complete()

        # Verify the log was copied
        expected_path = (
            fxt_training_job.data_dir
            / "projects"
            / str(fxt_training_job.project_id)
            / "models"
            / str(fxt_training_job.params.model_id)
            / "training.log"
        )
        assert expected_path.exists()
        assert expected_path.read_text() == "Training log content"

    @patch("app.models.jobs.training_job.logger")
    def test_on_complete_logs_warning(self, mock_logger, fxt_training_job):
        """Test that a warning is logged and no file copied when the source log file doesn't exist."""
        # Don't create the log file

        # Execute
        fxt_training_job.on_complete()

        # Verify warning was logged and no file was copied
        log_path = fxt_training_job.log_dir / fxt_training_job.log_file
        mock_logger.warning.assert_called_once_with(f"Log file {log_path} does not exist")
        expected_path = (
            fxt_training_job.data_dir
            / "projects"
            / str(fxt_training_job.project_id)
            / "models"
            / str(fxt_training_job.params.model_id)
            / "training.log"
        )
        assert not expected_path.exists()

    def test_on_complete_removes_getitune_workspace(self, fxt_training_job):
        """Test that the getitune workspace directory is removed on job completion."""
        # Create the workspace directory with a timestamped subdir and a stray file
        workspace_dir = fxt_training_job.data_dir / f"getitune-workspace-{fxt_training_job.params.model_id}"
        timestamp_dir = workspace_dir / "20260101_000000"
        timestamp_dir.mkdir(parents=True)
        (timestamp_dir / "leftover.txt").write_text("temp")

        fxt_training_job.on_complete()

        assert not workspace_dir.exists()

    def test_on_complete_no_op_when_workspace_missing(self, fxt_training_job):
        """on_complete must not raise when the getitune workspace was never created."""
        workspace_dir = fxt_training_job.data_dir / f"getitune-workspace-{fxt_training_job.params.model_id}"
        assert not workspace_dir.exists()

        # Should not raise
        fxt_training_job.on_complete()


class TestTrainingJobReconcileTrainingStatus:
    @patch("app.services.ModelService")
    @patch("app.db.engine.get_db_session")
    def test_forces_failed_when_status_stuck_in_progress(
        self, mock_get_db_session, mock_model_service_cls, fxt_training_job
    ):
        """A model left at IN_PROGRESS after job completion (e.g. a trainer crash) must be forced to FAILED."""
        mock_get_db_session.return_value.__enter__.return_value = Mock()
        mock_model_service = mock_model_service_cls.return_value
        mock_model_service.get_model.return_value = Mock(training_info=Mock(status=TrainingStatus.IN_PROGRESS))

        fxt_training_job._reconcile_stuck_training_status()

        mock_model_service.update_revision_status.assert_called_once()
        call_kwargs = mock_model_service.update_revision_status.call_args.kwargs
        assert call_kwargs["project_id"] == fxt_training_job.project_id
        assert call_kwargs["model_id"] == fxt_training_job.params.model_id
        assert call_kwargs["training_status"] == TrainingStatus.FAILED

    @pytest.mark.parametrize("terminal_status", [TrainingStatus.SUCCESSFUL, TrainingStatus.FAILED])
    @patch("app.services.ModelService")
    @patch("app.db.engine.get_db_session")
    def test_no_op_when_status_already_terminal(
        self, mock_get_db_session, mock_model_service_cls, terminal_status, fxt_training_job
    ):
        """A model that already reached a terminal status must not be touched."""
        mock_get_db_session.return_value.__enter__.return_value = Mock()
        mock_model_service = mock_model_service_cls.return_value
        mock_model_service.get_model.return_value = Mock(training_info=Mock(status=terminal_status))

        fxt_training_job._reconcile_stuck_training_status()

        mock_model_service.update_revision_status.assert_not_called()

    @patch("app.services.ModelService")
    @patch("app.db.engine.get_db_session")
    def test_no_op_when_model_revision_missing(self, mock_get_db_session, mock_model_service_cls, fxt_training_job):
        """A job that failed before the model revision was ever created has nothing to reconcile."""
        mock_get_db_session.return_value.__enter__.return_value = Mock()
        mock_model_service = mock_model_service_cls.return_value
        mock_model_service.get_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, "missing")

        fxt_training_job._reconcile_stuck_training_status()

        mock_model_service.update_revision_status.assert_not_called()
