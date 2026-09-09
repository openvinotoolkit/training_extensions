# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from enum import StrEnum
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.db.schema import ModelVariantDB, PipelineDB, ProjectDB
from app.models import DataCollectionConfig, InferenceConfig, PipelineStatus
from app.models.data_collection_policy import FixedRateDataCollectionPolicy
from app.models.model_revision import ModelFormat, ModelPrecision, TrainingStatus
from app.models.system import DeviceInfo, DeviceType
from app.services import ResourceNotFoundError, ResourceType
from app.services.event.event_bus import EventType
from app.services.pipeline_service import (
    DeviceInt8NotSupportedError,
    FolderSinkNotAccessibleError,
    IncompatibleModelVariantError,
    OtherProjectActiveError,
)
from tests.integration.model_files import write_openvino_model
from tests.integration.project_factory import ProjectTestDataFactory


class PipelineField(StrEnum):
    SOURCE_ID = "source_id"
    SINK_ID = "sink_id"


@pytest.fixture
def fxt_project_with_pipeline(
    fxt_db_projects, fxt_db_sinks, fxt_db_sources, fxt_db_models, fxt_db_model_variants, db_session
) -> Callable[[bool, int, list[dict] | None, str], tuple[ProjectDB, PipelineDB]]:
    """Fixture to create a ProjectDB with an associated PipelineDB, including model variants."""

    def _create_project_with_pipeline(
        is_running: bool, project_index: int = 0, data_policies: list[dict] | None = None, device: str = "cpu"
    ) -> tuple[ProjectDB, PipelineDB]:
        db_session.add_all(fxt_db_sources)
        db_session.add_all(fxt_db_sinks)
        db_session.flush()
        # Build project with pipeline (without model_variant_id, since variants need models to exist first)
        project_db = (
            ProjectTestDataFactory(db_session)
            .with_project(fxt_db_projects[project_index])
            .with_pipeline(
                is_running=is_running,
                model_id=fxt_db_models[project_index].id,
                source_id=fxt_db_sources[project_index].id,
                sink_id=fxt_db_sinks[project_index].id,
                device=device,
            )
            .with_models(fxt_db_models)
            .with_data_policies(data_policies if data_policies else [])
            .build()
        )
        # Persist model variants (requires model revisions to already exist)
        db_session.add_all(fxt_db_model_variants)
        db_session.flush()
        # Now set the model_variant_id FK on the pipeline
        pipeline_db = db_session.get(PipelineDB, project_db.id)
        pipeline_db.model_variant_id = fxt_db_model_variants[project_index].id
        db_session.flush()
        return project_db, db_session.get(PipelineDB, project_db.id)

    return _create_project_with_pipeline


class TestPipelineServiceIntegration:
    """Integration tests for PipelineService."""

    def test_create_pipeline(self, fxt_pipeline_service, fxt_project_id, fxt_db_projects, db_session):
        """Test creating a pipeline."""
        (ProjectTestDataFactory(db_session).with_project(fxt_db_projects[0]).build())

        pipeline = fxt_pipeline_service.create_pipeline(fxt_project_id)

        assert (
            pipeline is not None
            and str(pipeline.project_id) == str(fxt_project_id)
            and pipeline.source_id is None
            and pipeline.source is None
            and pipeline.sink_id is None
            and pipeline.sink is None
            and pipeline.model_id is None
            and pipeline.model_revision is None
            and pipeline.status == PipelineStatus.IDLE
            and pipeline.data_collection == DataCollectionConfig()
        )

    def test_get_pipeline(self, fxt_pipeline_service, fxt_project_id, fxt_project_with_pipeline, db_session):
        """Test retrieving a pipeline by ID."""
        _, db_pipeline = fxt_project_with_pipeline(
            is_running=False, data_policies=[{"type": "fixed_rate", "enabled": True, "rate": 0.1}]
        )

        pipeline = fxt_pipeline_service.get_pipeline_by_id(fxt_project_id)

        assert pipeline is not None
        assert pipeline.project_id == fxt_project_id
        assert pipeline.status == PipelineStatus.IDLE
        assert pipeline.sink.name == db_pipeline.sink.name
        assert pipeline.source.name == db_pipeline.source.name
        assert str(pipeline.model_id) == db_pipeline.model_revision_id
        assert pipeline.data_collection.policies == [FixedRateDataCollectionPolicy(rate=0.1)]

    def test_get_active_pipeline(self, fxt_pipeline_service, fxt_project_with_pipeline, db_session):
        """Test retrieving a pipeline by ID."""
        db_project, _ = fxt_project_with_pipeline(is_running=True, data_policies=[])

        project_id = UUID(db_project.id)
        active_pipeline = fxt_pipeline_service.get_active_pipeline()

        assert active_pipeline is not None
        assert active_pipeline.project_id == project_id

    def test_get_active_pipeline_device_change(self, fxt_pipeline_service, fxt_project_with_pipeline, db_session):
        """Test retrieving a pipeline when its original device is no longer available."""
        db_project, db_pipeline = fxt_project_with_pipeline(is_running=True, data_policies=[], device="xpu-99")

        assert db_pipeline.device == "xpu-99"

        project_id = UUID(db_project.id)
        active_pipeline = fxt_pipeline_service.get_active_pipeline()

        assert active_pipeline is not None
        assert active_pipeline.project_id == project_id
        assert active_pipeline.device == "cpu"

    def test_get_non_existent_pipeline(self, fxt_pipeline_service):
        """Test retrieving a non-existent pipeline raises error."""
        pipeline_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_pipeline_service.get_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)

    def test_update_pipeline_raises_error_if_other_project_active(
        self,
        fxt_pipeline_service,
        fxt_project_with_pipeline,
    ):
        """Test that updating a pipeline to running raises error if another project's pipeline is already running."""
        # Create an active pipeline in one project
        active_project_db, active_pipeline = fxt_project_with_pipeline(is_running=True, project_index=0)
        # Create a second project/pipeline (not running)
        _, other_pipeline = fxt_project_with_pipeline(is_running=False, project_index=1)
        # Try to activate the second pipeline while the first is still running
        with pytest.raises(OtherProjectActiveError) as excinfo:
            fxt_pipeline_service.update_pipeline(other_pipeline.project_id, {"status": PipelineStatus.RUNNING})
        assert active_project_db.name in str(excinfo.value)
        assert str(active_pipeline.project_id) not in str(excinfo.value)

    @pytest.mark.parametrize("pipeline_attr", [PipelineField.SINK_ID, PipelineField.SOURCE_ID])
    def test_reconfigure_running_pipeline(
        self,
        pipeline_attr,
        fxt_project_with_pipeline,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_pipeline_service,
        fxt_event_bus,
        db_session,
    ):
        """Test updating a pipeline by ID."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        if pipeline_attr == PipelineField.SINK_ID:
            item_id = fxt_db_sinks[1].id
        else:
            item_id = fxt_db_sources[1].id

        updated = fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {pipeline_attr: item_id})

        if pipeline_attr == PipelineField.SINK_ID:
            fxt_event_bus.emit_event_after_commit.assert_called_once_with(db_session, EventType.SINK_CHANGED)
        else:
            fxt_event_bus.emit_event_after_commit.assert_called_once_with(db_session, EventType.SOURCE_CHANGED)
        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        assert str(getattr(updated, pipeline_attr)) == item_id
        assert str(getattr(updated, pipeline_attr)) == getattr(db_updated, pipeline_attr)

    @pytest.mark.parametrize("model_attr", ["model_id", "model_revision_id"])
    def test_switch_model(
        self,
        model_attr,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_db_model_variants,
        fxt_pipeline_service,
        fxt_event_bus,
        db_session,
    ):
        """Test switching the model on a pipeline defaults to the FP16 OpenVINO variant."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        model_id = fxt_db_models[1].id

        updated = fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {model_attr: model_id})

        fxt_event_bus.emit_event_after_commit.assert_called_once_with(db_session, EventType.MODEL_CHANGED)
        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        assert str(updated.model_id) == model_id
        assert str(updated.model_id) == db_updated.model_revision_id
        # Default FP16 OpenVINO variant should have been resolved
        assert db_updated.model_variant_id == fxt_db_model_variants[1].id

    def test_switch_model_with_explicit_variant(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_db_model_variants,
        fxt_pipeline_service,
        fxt_event_bus,
        db_session,
    ):
        """Test switching model with an explicit model_variant_id selects that variant."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[1]
        target_variant = fxt_db_model_variants[1]

        updated = fxt_pipeline_service.update_pipeline(
            db_pipeline.project_id,
            {"model_id": target_model.id, "model_variant_id": target_variant.id},
        )

        fxt_event_bus.emit_event_after_commit.assert_called_once_with(db_session, EventType.MODEL_CHANGED)
        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        assert str(updated.model_id) == target_model.id
        assert db_updated.model_variant_id == target_variant.id

    def test_switch_variant_only_derives_model_id(
        self,
        fxt_project_with_pipeline,
        fxt_pipeline_service,
    ):
        """
        Providing only `model_variant_id` (no `model_id`) should raise an error
        """
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        with pytest.raises(ValueError, match="It is not possible to provide only a model variant ID"):
            _ = fxt_pipeline_service.update_pipeline(
                db_pipeline.project_id,
                {"model_variant_id": str(uuid4())},
            )

    def test_switch_model_defaults_to_fp16_openvino_variant(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_db_model_variants,
        fxt_pipeline_service,
        db_session,
    ):
        """Test that switching model without specifying a variant defaults to the FP16 OpenVINO variant."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[1]

        fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"model_id": target_model.id})

        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        assert db_updated.model_variant_id == fxt_db_model_variants[1].id

    def test_switch_model_rejects_non_openvino_variant(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_pipeline_service,
        db_session,
    ):
        """Test that specifying a non-OpenVINO variant raises IncompatibleModelVariantError."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[1]
        # Create a PyTorch variant for the target model
        pytorch_variant = ModelVariantDB(
            id=str(uuid4()),
            model_revision_id=target_model.id,
            format=ModelFormat.PYTORCH,
            precision=ModelPrecision.FP32,
        )
        db_session.add(pytorch_variant)
        db_session.flush()

        with pytest.raises(IncompatibleModelVariantError, match="Only OpenVINO model variants"):
            fxt_pipeline_service.update_pipeline(
                db_pipeline.project_id,
                {"model_id": target_model.id, "model_variant_id": pytorch_variant.id},
            )

    def test_switch_model_rejects_variant_from_wrong_revision(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_db_model_variants,
        fxt_pipeline_service,
        db_session,
    ):
        """
        Test that specifying a variant belonging to a different model revision raises IncompatibleModelVariantError.
        """
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[1]
        # Use a variant from the first model (index 0), which doesn't belong to target_model
        wrong_variant = fxt_db_model_variants[0]

        with pytest.raises(IncompatibleModelVariantError, match="does not belong to"):
            fxt_pipeline_service.update_pipeline(
                db_pipeline.project_id,
                {"model_id": target_model.id, "model_variant_id": wrong_variant.id},
            )

    def test_switch_model_raises_when_no_fp16_variant_exists(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_db_model_variants,
        fxt_pipeline_service,
        db_session,
    ):
        """Test that switching to a model with no FP16 OpenVINO variant raises IncompatibleModelVariantError."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[1]
        # Delete the FP16 variant by marking its files as deleted
        target_variant = fxt_db_model_variants[1]
        target_variant.files_deleted = True
        db_session.flush()

        with pytest.raises(IncompatibleModelVariantError, match="No FP16 OpenVINO variant found"):
            fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"model_id": target_model.id})

    def test_switch_model_rejects_variant_not_found(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_pipeline_service,
        db_session,
    ):
        """Test that specifying a non-existent variant ID raises ResourceNotFoundError."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[1]
        non_existent_variant_id = str(uuid4())

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_pipeline_service.update_pipeline(
                db_pipeline.project_id,
                {"model_id": target_model.id, "model_variant_id": non_existent_variant_id},
            )
        assert exc_info.value.resource_type == ResourceType.MODEL_VARIANT
        assert exc_info.value.resource_id == non_existent_variant_id

    def test_switch_model_int8_variant_raises_on_unsupported_device(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_pipeline_service,
        fxt_system_service,
        db_session,
    ):
        """
        Test that selecting an INT8 variant on a device that doesn't support INT8 raises DeviceInt8NotSupportedError.
        """
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[0]
        # Create an INT8 OpenVINO variant
        int8_variant = ModelVariantDB(
            id=str(uuid4()),
            model_revision_id=target_model.id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.INT8,
        )
        db_session.add(int8_variant)
        db_session.flush()

        # Mock the system service to report no INT8 support
        fxt_system_service.get_device_info = MagicMock(
            return_value=DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
        )
        fxt_system_service.supports_int8 = MagicMock(return_value=False)

        with pytest.raises(DeviceInt8NotSupportedError):
            fxt_pipeline_service.update_pipeline(
                db_pipeline.project_id,
                {"model_id": target_model.id, "model_variant_id": int8_variant.id},
            )

    def test_switch_model_int8_variant_succeeds_on_supported_device(
        self,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_pipeline_service,
        fxt_system_service,
        fxt_event_bus,
        db_session,
    ):
        """Test that selecting an INT8 variant on a device that supports INT8 succeeds."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        target_model = fxt_db_models[0]
        # Create an INT8 OpenVINO variant
        int8_variant = ModelVariantDB(
            id=str(uuid4()),
            model_revision_id=target_model.id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.INT8,
        )
        db_session.add(int8_variant)
        db_session.flush()

        # Mock the system service to report INT8 support
        fxt_system_service.get_device_info = MagicMock(
            return_value=DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
        )
        fxt_system_service.supports_int8 = MagicMock(return_value=True)

        fxt_pipeline_service.update_pipeline(
            db_pipeline.project_id,
            {"model_id": target_model.id, "model_variant_id": int8_variant.id},
        )

        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        assert db_updated.model_variant_id == int8_variant.id

    @pytest.mark.parametrize(
        "training_status",
        [TrainingStatus.NOT_STARTED, TrainingStatus.IN_PROGRESS, TrainingStatus.FAILED],
    )
    def test_update_model_raises_when_not_successfully_trained(
        self,
        training_status,
        fxt_project_with_pipeline,
        fxt_db_models,
        fxt_pipeline_service,
        db_session,
    ):
        """Test that updating the pipeline model raises ValueError when the target model is not successfully trained."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

        new_model = fxt_db_models[1]
        new_model.training_status = training_status
        db_session.flush()

        with pytest.raises(ValueError, match="points to a model that was not successfully trained"):
            fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"model_id": new_model.id})

        # Pipeline model should remain unchanged
        db_unchanged = db_session.get(PipelineDB, db_pipeline.project_id)
        assert db_unchanged.model_revision_id == db_pipeline.model_revision_id

    def test_update_model_raises_when_model_not_found(
        self,
        fxt_project_with_pipeline,
        fxt_pipeline_service,
        db_session,
    ):
        """Test that updating the pipeline model raises ResourceNotFoundError when the model ID does not exist."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)
        non_existent_model_id = str(uuid4())

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"model_id": non_existent_model_id})

        assert exc_info.value.resource_type == ResourceType.MODEL
        assert exc_info.value.resource_id == non_existent_model_id

    @pytest.mark.parametrize("pipeline_status", [PipelineStatus.IDLE, PipelineStatus.RUNNING])
    def test_enable_disable_pipeline(
        self,
        pipeline_status,
        fxt_project_with_pipeline,
        fxt_pipeline_service,
        fxt_event_bus,
        db_session,
    ):
        """Test activating and deactivating an existing pipeline by ID."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=not pipeline_status.as_bool)

        fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"status": pipeline_status})

        fxt_event_bus.emit_event_after_commit.assert_called_once_with(db_session, EventType.PIPELINE_STATUS_CHANGED)
        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        if pipeline_status == PipelineStatus.RUNNING:
            assert db_updated.is_running
        else:
            assert not db_updated.is_running

    @pytest.mark.parametrize("config", ["source", "model_revision"])
    def test_enable_misconfigured_pipeline(self, config, fxt_project_with_pipeline, fxt_pipeline_service, db_session):
        """Test enabling a misconfigured pipeline raises error.

        Note: a sink is intentionally NOT required to enable a pipeline. When no sink is configured,
        predictions are still routed to the WebRTC visualization stream.
        """
        _, db_pipeline = fxt_project_with_pipeline(is_running=False)
        setattr(db_pipeline, config, None)  # Misconfigure the pipeline
        db_session.flush()

        with pytest.raises(ValueError, match="Pipeline cannot be in 'running' state"):
            fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"status": PipelineStatus.RUNNING})

        assert not db_session.get(PipelineDB, db_pipeline.project_id).is_running

    def test_enable_pipeline_with_inaccessible_folder_sink(
        self,
        fxt_project_with_pipeline,
        fxt_pipeline_service,
        db_session,
        tmp_path,
    ):
        """
        Test that enabling a pipeline with a folder sink that cannot be created raises FolderSinkNotAccessibleError.
        """
        _, db_pipeline = fxt_project_with_pipeline(is_running=False)

        # Create a file, then try to use it as a parent directory - always fails regardless of privileges
        blocking_file = tmp_path / "not_a_directory"
        blocking_file.write_bytes(b"")
        inaccessible_path = str(blocking_file / "subdir")

        db_pipeline.sink.config_data = {"folder_path": inaccessible_path}
        db_session.flush()

        with pytest.raises(FolderSinkNotAccessibleError):
            fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"status": PipelineStatus.RUNNING})

        assert not db_session.get(PipelineDB, db_pipeline.project_id).is_running

    def test_enable_pipeline_without_sink(self, fxt_project_with_pipeline, fxt_pipeline_service, db_session):
        """A pipeline can be enabled even when no sink is configured."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=False)
        db_pipeline.sink = None
        db_pipeline.sink_id = None
        db_session.flush()

        updated = fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"status": PipelineStatus.RUNNING})

        assert updated.status == PipelineStatus.RUNNING
        assert updated.sink_id is None
        assert db_session.get(PipelineDB, db_pipeline.project_id).is_running

    def test_reconfigure_non_existent_pipeline(self, fxt_pipeline_service):
        """Test updating a non-existent pipeline raises error."""
        project_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_pipeline_service.update_pipeline(project_id, {"sink_id": uuid4()})

        assert exc_info.value.resource_type == ResourceType.PIPELINE
        assert exc_info.value.resource_id == str(project_id)

    def test_set_pipeline_data_collection_config(
        self,
        fxt_pipeline_service,
        fxt_project_id,
        fxt_db_projects,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_db_models,
        fxt_db_model_variants,
        db_session,
    ):
        """Test setting pipeline data collection config."""
        db_session.add_all([fxt_db_sinks[0], fxt_db_sources[0]])
        db_session.flush()
        project_db = (
            ProjectTestDataFactory(db_session)
            .with_project(fxt_db_projects[0])
            .with_pipeline(
                sink_id=fxt_db_sinks[0].id,
                source_id=fxt_db_sources[0].id,
                model_id=fxt_db_models[0].id,
            )
            .with_models([fxt_db_models[0]])
            .build()
        )
        db_session.add(fxt_db_model_variants[0])
        db_session.flush()
        pipeline_db = db_session.get(PipelineDB, project_db.id)
        pipeline_db.model_variant_id = fxt_db_model_variants[0].id
        db_session.flush()

        pipeline = fxt_pipeline_service.update_pipeline(
            project_id=fxt_project_id,
            partial_config={
                "data_collection": DataCollectionConfig(
                    max_dataset_size=500,
                    policies=[FixedRateDataCollectionPolicy(type="fixed_rate", rate=0.1)],
                )
            },
        )

        assert pipeline is not None
        assert pipeline.data_collection.max_dataset_size == 500
        assert pipeline.data_collection.policies == [FixedRateDataCollectionPolicy(type="fixed_rate", rate=0.1)]

    def test_reset_pipeline_data_collection_config(
        self,
        fxt_pipeline_service,
        fxt_project_id,
        fxt_db_projects,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_db_models,
        fxt_db_model_variants,
        db_session,
    ):
        """Test resetting pipeline data collection config."""
        db_session.add_all([fxt_db_sinks[0], fxt_db_sources[0]])
        db_session.flush()
        project_db = (
            ProjectTestDataFactory(db_session)
            .with_project(fxt_db_projects[0])
            .with_pipeline(
                sink_id=fxt_db_sinks[0].id,
                source_id=fxt_db_sources[0].id,
                model_id=fxt_db_models[0].id,
            )
            .with_models([fxt_db_models[0]])
            .with_data_policies([{"type": "fixed_rate", "enabled": True, "rate": 0.1}])
            .build()
        )
        db_session.add(fxt_db_model_variants[0])
        db_session.flush()
        pipeline_db = db_session.get(PipelineDB, project_db.id)
        pipeline_db.model_variant_id = fxt_db_model_variants[0].id
        db_session.flush()

        pipeline = fxt_pipeline_service.update_pipeline(
            project_id=fxt_project_id, partial_config={"data_collection": DataCollectionConfig()}
        )

        assert pipeline is not None
        assert pipeline.data_collection == DataCollectionConfig()


class TestPipelineConfidenceThreshold:
    """Integration tests for the inference confidence threshold of a pipeline."""

    @pytest.fixture
    def fxt_pipeline_with_models(
        self,
        fxt_project_with_pipeline,
        fxt_project_id,
        fxt_db_models,
        fxt_db_model_variants,
        fxt_projects_dir,
    ):
        """Set up a pipeline whose models embed the given confidence thresholds, one per model."""

        def _create(thresholds: list[float | None], is_running: bool = True) -> None:
            fxt_project_with_pipeline(is_running=is_running)
            for model, variant, threshold in zip(fxt_db_models, fxt_db_model_variants, thresholds):
                write_openvino_model(
                    fxt_projects_dir / str(fxt_project_id) / "models" / model.id / "variants" / variant.id,
                    confidence_threshold=threshold,
                )

        return _create

    @pytest.mark.parametrize("embedded_threshold", [0.35, None], ids=["with_threshold", "without_threshold"])
    def test_get_pipeline_reports_model_confidence_threshold(
        self, embedded_threshold, fxt_pipeline_with_models, fxt_pipeline_service, fxt_project_id
    ):
        """A pipeline that never overrode the threshold reports the one embedded in the model, if any."""
        fxt_pipeline_with_models([embedded_threshold, None], is_running=False)

        pipeline = fxt_pipeline_service.get_pipeline_by_id(fxt_project_id)

        assert pipeline.inference.confidence_threshold == pytest.approx(embedded_threshold)

    def test_get_pipeline_without_model_reports_no_threshold(
        self, fxt_pipeline_service, fxt_project_id, fxt_db_projects, db_session
    ):
        """A pipeline without a model has no confidence threshold to report."""
        ProjectTestDataFactory(db_session).with_project(fxt_db_projects[0]).with_pipeline().build()

        pipeline = fxt_pipeline_service.get_pipeline_by_id(fxt_project_id)

        assert pipeline.inference.confidence_threshold is None

    @pytest.mark.parametrize("embedded_threshold", [0.35, None], ids=["with_threshold", "without_threshold"])
    def test_get_pipeline_reports_model_variant_confidence_threshold(
        self, embedded_threshold, fxt_pipeline_with_models, fxt_pipeline_service, fxt_project_id
    ):
        """The embedded model variant returned by the pipeline also reports the confidence threshold."""
        fxt_pipeline_with_models([embedded_threshold, None], is_running=False)

        pipeline = fxt_pipeline_service.get_pipeline_by_id(fxt_project_id)

        assert pipeline.model_variant is not None
        assert pipeline.model_variant.optimal_confidence_threshold == pytest.approx(embedded_threshold)

    def test_get_pipeline_model_variant_threshold_independent_of_custom_override(
        self, fxt_pipeline_with_models, fxt_pipeline_service, fxt_project_id
    ):
        """The model variant's threshold always reflects the model file, not a custom pipeline override."""
        fxt_pipeline_with_models([0.35, None], is_running=False)
        fxt_pipeline_service.update_pipeline(
            str(fxt_project_id), {"inference": InferenceConfig(confidence_threshold=0.7)}
        )

        pipeline = fxt_pipeline_service.get_pipeline_by_id(fxt_project_id)

        assert pipeline.inference.confidence_threshold == pytest.approx(0.7)
        assert pipeline.model_variant is not None
        assert pipeline.model_variant.optimal_confidence_threshold == pytest.approx(0.35)

    def test_set_custom_confidence_threshold(
        self, fxt_pipeline_with_models, fxt_pipeline_service, fxt_project_id, fxt_event_bus, db_session
    ):
        """A custom threshold is persisted and applied to the running model without reloading it."""
        fxt_pipeline_with_models([0.35, 0.6])

        updated = fxt_pipeline_service.update_pipeline(
            str(fxt_project_id), {"inference": InferenceConfig(confidence_threshold=0.7)}
        )

        assert updated.inference.confidence_threshold == pytest.approx(0.7)
        assert db_session.get(PipelineDB, str(fxt_project_id)).inference == {"confidence_threshold": 0.7}
        fxt_event_bus.emit_event_after_commit.assert_called_once_with(db_session, EventType.INFERENCE_PARAMS_CHANGED)

    @pytest.mark.parametrize(
        "follow_up_config, expected_threshold",
        [
            pytest.param(lambda models: {"inference": InferenceConfig()}, 0.35, id="explicit_null_restores_model"),
            pytest.param(lambda models: {"model_id": models[1].id}, 0.6, id="model_switch_resets_to_new_model"),
            pytest.param(
                lambda models: {"data_collection": DataCollectionConfig(max_dataset_size=10)},
                0.7,
                id="unrelated_update_keeps_custom",
            ),
        ],
    )
    def test_update_after_custom_confidence_threshold(
        self,
        follow_up_config,
        expected_threshold,
        fxt_pipeline_with_models,
        fxt_pipeline_service,
        fxt_project_id,
        fxt_db_models,
    ):
        """A custom threshold only stays in effect while the pipeline keeps using the same model."""
        fxt_pipeline_with_models([0.35, 0.6])
        fxt_pipeline_service.update_pipeline(
            str(fxt_project_id), {"inference": InferenceConfig(confidence_threshold=0.7)}
        )

        updated = fxt_pipeline_service.update_pipeline(str(fxt_project_id), follow_up_config(fxt_db_models))

        assert updated.inference.confidence_threshold == pytest.approx(expected_threshold)

    def test_switch_model_with_explicit_threshold(
        self, fxt_pipeline_with_models, fxt_pipeline_service, fxt_project_id, fxt_db_models
    ):
        """A threshold provided together with a new model takes precedence over the model's own value."""
        fxt_pipeline_with_models([0.35, 0.6])

        updated = fxt_pipeline_service.update_pipeline(
            str(fxt_project_id),
            {"model_id": fxt_db_models[1].id, "inference": InferenceConfig(confidence_threshold=0.8)},
        )

        assert updated.inference.confidence_threshold == pytest.approx(0.8)
