# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.db.schema import LabelDB, ModelRevisionDB, ModelVariantDB, PipelineDB, ProjectDB
from app.models.model_revision import ModelFormat, ModelPrecision, TrainingStatus
from app.models.system import DeviceInfo, DeviceType
from app.services import ActiveModelService, SystemService
from app.services.inference.model_loader import LoadedModelHandle, ModelLoader
from tests.integration.project_factory import ProjectTestDataFactory


@pytest.fixture
def fxt_project(fxt_db_projects, db_session) -> ProjectDB:
    """Fixture to create a project in the database."""
    return ProjectTestDataFactory(db_session).with_project(fxt_db_projects[0]).build()


@pytest.fixture
def fxt_successful_model(fxt_project) -> ModelRevisionDB:
    """Fixture providing a successfully trained model revision."""
    return ModelRevisionDB(
        id=str(uuid4()),
        project_id=fxt_project.id,
        name="YOLOX-S (successful)",
        architecture="object-detection-yolox-s",
        training_status=TrainingStatus.SUCCESSFUL,
        training_configuration={},
        label_schema_revision={},
    )


@pytest.fixture
def fxt_failed_model(fxt_project) -> ModelRevisionDB:
    """Fixture providing a failed model revision."""
    return ModelRevisionDB(
        id=str(uuid4()),
        project_id=fxt_project.id,
        name="YOLOX-S (failed)",
        architecture="object-detection-yolox-s",
        training_status=TrainingStatus.FAILED,
        training_configuration={},
        label_schema_revision={},
    )


@pytest.fixture
def fxt_in_progress_model(fxt_project) -> ModelRevisionDB:
    """Fixture providing an in-progress model revision."""
    return ModelRevisionDB(
        id=str(uuid4()),
        project_id=fxt_project.id,
        name="YOLOX-S (in_progress)",
        architecture="object-detection-yolox-s",
        training_status=TrainingStatus.IN_PROGRESS,
        training_configuration={},
        label_schema_revision={},
    )


@pytest.fixture
def fxt_fp16_openvino_variant(fxt_successful_model) -> ModelVariantDB:
    """Fixture providing an FP16 OpenVINO model variant for the successful model."""
    return ModelVariantDB(
        id=str(uuid4()),
        model_revision_id=fxt_successful_model.id,
        format=ModelFormat.OPENVINO,
        precision=ModelPrecision.FP16,
    )


def _make_db_session_patcher(db_session):
    """Return a context manager that patches get_db_session to yield the test db_session."""

    @contextmanager
    def _patched_get_db_session():
        yield db_session

    return patch("app.services.active_model_service.get_db_session", side_effect=_patched_get_db_session)


def _fake_handle() -> tuple[LoadedModelHandle, MagicMock]:
    """A loaded model handle, plus the mock model that records the parameters pushed onto it."""
    model = MagicMock()
    handle = LoadedModelHandle(
        model_id=uuid4(),
        variant_id=uuid4(),
        device=DeviceInfo(type=DeviceType.CPU, name="CPU"),
        model=model,
        loaded_at=datetime.now(),
    )
    return handle, model


class TestActiveModelServiceLoadState:
    """Integration tests for ActiveModelService._load_state."""

    def test_load_state_no_active_pipeline_returns_empty_state(self, db_session, tmp_path):
        """When no pipeline is running, load_state returns an empty ModelActivationState."""
        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path, system_service=SystemService())

        state = service._model_activation_state

        assert state.project_id is None
        assert state.active_model_id is None
        assert state.active_model_variant_id is None
        assert state.available_models == []

    def test_load_state_only_includes_successful_models(
        self,
        fxt_project,
        fxt_successful_model,
        fxt_failed_model,
        fxt_in_progress_model,
        fxt_fp16_openvino_variant,
        db_session,
        tmp_path,
    ):
        """available_models must only contain successfully trained model revisions."""
        # Persist models with different statuses and the active variant
        second_successful = ModelRevisionDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            name="YOLOX-X (successful)",
            architecture="object-detection-yolox-x",
            training_status=TrainingStatus.SUCCESSFUL,
            training_configuration={},
            label_schema_revision={},
        )
        db_session.add_all([fxt_successful_model, second_successful, fxt_failed_model, fxt_in_progress_model])
        db_session.add(fxt_fp16_openvino_variant)
        db_session.flush()

        # Create a running pipeline pointing at the successful model
        pipeline = PipelineDB(
            project_id=fxt_project.id,
            is_running=True,
            model_revision_id=fxt_successful_model.id,
            device="cpu",
        )
        db_session.add(pipeline)
        db_session.flush()

        system_service = SystemService()

        with (
            _make_db_session_patcher(db_session),
            patch.object(
                system_service, "inference_device", wraps=system_service.inference_device
            ) as mock_inference_device,
        ):
            service = ActiveModelService(data_dir=tmp_path, system_service=system_service)

        mock_inference_device.assert_called_once_with("cpu", fallback_to_cpu=True)

        state = service._model_activation_state
        assert len(state.available_models) == 2
        assert set(state.available_models) == {UUID(fxt_successful_model.id), UUID(second_successful.id)}
        assert str(state.active_model_id) == fxt_successful_model.id
        assert str(state.active_model_variant_id) == fxt_fp16_openvino_variant.id


class TestActiveModelServiceConfidenceThreshold:
    """Integration tests for applying the pipeline confidence threshold to the loaded model."""

    @pytest.fixture
    def fxt_running_pipeline(self, fxt_project, fxt_successful_model, fxt_fp16_openvino_variant, db_session, tmp_path):
        """Create a running pipeline with a successfully trained model and its FP16 OpenVINO variant."""

        def _create(inference: dict) -> PipelineDB:
            db_session.add_all([fxt_successful_model, fxt_fp16_openvino_variant])
            db_session.flush()
            pipeline = PipelineDB(
                project_id=fxt_project.id,
                is_running=True,
                model_revision_id=fxt_successful_model.id,
                model_variant_id=fxt_fp16_openvino_variant.id,
                device="cpu",
                inference=inference,
            )
            db_session.add(pipeline)
            db_session.flush()
            variant_dir = (
                tmp_path
                / "projects"
                / fxt_project.id
                / "models"
                / fxt_successful_model.id
                / "variants"
                / fxt_fp16_openvino_variant.id
            )
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "model.xml").touch()
            (variant_dir / "model.bin").touch()
            return pipeline

        return _create

    def test_configured_threshold_is_applied_on_load(self, fxt_running_pipeline, db_session, tmp_path):
        """A model loaded for a pipeline with a custom threshold gets that threshold pushed onto it."""
        fxt_running_pipeline({"confidence_threshold": 0.7})
        handle, model = _fake_handle()

        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path, system_service=SystemService())
            with patch.object(ModelLoader, "load", return_value=handle) as mock_load:
                service.get_loaded_inference_model()

        mock_load.assert_called_once()
        model.set_param.assert_called_once_with("confidence_threshold", 0.7)

    def test_model_value_is_kept_when_no_threshold_configured(self, fxt_running_pipeline, db_session, tmp_path):
        """Without a configured threshold, the value embedded in the model files is left untouched."""
        fxt_running_pipeline({})
        handle, model = _fake_handle()

        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path, system_service=SystemService())
            with patch.object(ModelLoader, "load", return_value=handle):
                service.get_loaded_inference_model()

        model.set_param.assert_not_called()

    def test_refresh_inference_params_updates_loaded_model(self, fxt_running_pipeline, db_session, tmp_path):
        """A threshold change is applied to the already-loaded model, without reloading it."""
        pipeline = fxt_running_pipeline({"confidence_threshold": 0.7})
        handle, model = _fake_handle()

        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path, system_service=SystemService())
            with patch.object(ModelLoader, "load", return_value=handle) as mock_load:
                service.get_loaded_inference_model()
                pipeline.inference = {"confidence_threshold": 0.2}
                db_session.flush()
                service.refresh_inference_params()

        mock_load.assert_called_once()  # the model was not reloaded
        assert model.set_param.call_args_list[-1].args == ("confidence_threshold", 0.2)


class TestActiveModelServiceLabelColors:
    """The colors of the project labels must be exposed so that the predictions can be
    rendered with the same colors as the labels defined in the project."""

    @pytest.fixture
    def fxt_pipeline_with_labels(self, fxt_project, fxt_successful_model, fxt_fp16_openvino_variant, db_session):
        def _create(labels: dict[str, str]) -> list[LabelDB]:
            db_labels = [
                LabelDB(id=str(uuid4()), project_id=fxt_project.id, name=name, color=color)
                for name, color in labels.items()
            ]
            db_session.add_all([fxt_successful_model, fxt_fp16_openvino_variant, *db_labels])
            db_session.flush()
            db_session.add(
                PipelineDB(
                    project_id=fxt_project.id,
                    is_running=True,
                    model_revision_id=fxt_successful_model.id,
                    model_variant_id=fxt_fp16_openvino_variant.id,
                    device="cpu",
                )
            )
            db_session.flush()
            return db_labels

        return _create

    def test_label_colors_of_the_active_project_are_loaded(self, fxt_pipeline_with_labels, db_session, tmp_path):
        fxt_pipeline_with_labels({"cat": "#00FF00", "dog": "#FF0000"})

        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path, system_service=SystemService())

        assert service.label_colors == {"cat": "#00FF00", "dog": "#FF0000"}

    def test_label_colors_are_empty_without_active_model(self, db_session, tmp_path):
        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path, system_service=SystemService())

        assert service.label_colors == {}

    def test_recolored_labels_are_picked_up_on_refresh(self, fxt_pipeline_with_labels, db_session, tmp_path):
        """Recoloring a label in the project is reflected without reloading the model."""
        db_labels = fxt_pipeline_with_labels({"cat": "#00FF00"})

        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path, system_service=SystemService())
            db_labels[0].color = "#123456"
            db_session.flush()
            service.refresh_inference_params()

        assert service.label_colors == {"cat": "#123456"}
