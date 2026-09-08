# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.db.engine import get_db_session
from app.models.model_activation import ModelActivationState
from app.models.model_revision import ModelFormat, ModelPrecision, TrainingStatus
from app.models.system import DeviceInfo, DeviceType
from app.repositories import ModelRevisionRepository, ModelVariantRepository
from app.repositories.active_model_repo import ActiveModelRepo
from app.repositories.label_repo import LabelRepository
from app.services.inference.model_loader import LoadedModelHandle, ModelLoader

from .system_service import SystemService


class ActiveModelService:
    """
    Service to fetch the currently active model for inference.

    Used exclusively by the InferenceWorker process.
    """

    def __init__(self, data_dir: Path, system_service: SystemService) -> None:
        self.projects_dir = data_dir / "projects"
        self._system_service = system_service
        self._model_activation_state: ModelActivationState = self._load_state()
        self._loaded_model: LoadedModelHandle | None = None

    def _load_state(self) -> ModelActivationState:
        """Load the state from the DB if it exists, otherwise initialize an empty state"""
        with get_db_session() as db:
            active_model_repo = ActiveModelRepo(db=db)
            active_model = active_model_repo.get_active_revision()
            if active_model is None:
                return ModelActivationState(
                    project_id=None,
                    active_model_id=None,
                    active_model_variant_id=None,
                    available_models=[],
                    device=DeviceInfo(type=DeviceType.CPU, name="cpu"),
                )
            model_rev_repo = ModelRevisionRepository(project_id=str(active_model.project_id), db=db)
            available_models = model_rev_repo.list_all(training_status=TrainingStatus.SUCCESSFUL)
            # Use the variant configured in the pipeline, fall back to FP16 OpenVINO
            active_variant_id = active_model_repo.get_active_model_variant_id()
            if active_variant_id is None:
                model_variants_repo = ModelVariantRepository(db=db)
                model_variants = model_variants_repo.list_by_model_revision(str(active_model.id))
                active_variant_id = next(
                    v.id
                    for v in model_variants
                    if v.format == ModelFormat.OPENVINO and v.precision == ModelPrecision.FP16
                )
                logger.warning("No active model variant ID found, loaded fallback model %s", active_variant_id)
            pipeline_device = active_model_repo.get_active_pipeline_device()
            if pipeline_device is None:
                raise RuntimeError("Active pipeline must have a device configured")
            geti_device = self._system_service.inference_device(pipeline_device, fallback_to_cpu=True)
            return ModelActivationState(
                project_id=UUID(active_model.project_id),
                active_model_id=UUID(active_model.id),
                active_model_variant_id=UUID(active_variant_id),
                available_models=[UUID(m.id) for m in available_models],
                device=geti_device,
                confidence_threshold=active_model_repo.get_active_confidence_threshold(),
                label_colors=self._load_label_colors(db=db, project_id=str(active_model.project_id)),
            )

    @staticmethod
    def _load_label_colors(db: Session, project_id: str) -> dict[str, str]:
        """Load the mapping of label name to label colour for the given project.

        The predictions returned by Model API carry the label names the model was trained on,
        which are the project label names. Mapping them back to the project colours ensures the
        rendered predictions use the same colours as the project labels shown in the UI.
        """
        try:
            labels = LabelRepository(project_id=project_id, db=db).list_all()
        except Exception:
            logger.exception("Failed to load label colors for project '{}'", project_id)
            return {}
        return {label.name: label.color for label in labels if label.name and label.color}

    @property
    def label_colors(self) -> dict[str, str]:
        """Mapping of label name to hex colour for the project owning the active model."""
        return self._model_activation_state.label_colors

    def _get_model_file_path(self, project_id: UUID, model_id: UUID, variant_id: UUID, extension: str = "xml") -> Path:
        file_path = self.projects_dir / f"{project_id}/models/{model_id}/variants/{variant_id}/model.{extension}"
        if file_path.is_file():
            return file_path
        raise FileNotFoundError(f"Model file not found: {file_path}")

    def get_loaded_inference_model(self, force_reload: bool = False) -> LoadedModelHandle | None:
        """
        Get the currently active model for inference.

        Args:
            force_reload: If True, reload the state and the model from disk. This option can be useful
            to bypass the cache after the state has been modified externally.

        Returns: Model for inference or None if no model is active, or if the model can't be loaded.
        """
        if force_reload:
            self._unload_model()
            self._model_activation_state = self._load_state()

        if (
            self._model_activation_state.active_model_id is None
            or self._model_activation_state.active_model_variant_id is None
            or self._model_activation_state.project_id is None
        ):
            return None

        project_id = self._model_activation_state.project_id
        active_model_id = self._model_activation_state.active_model_id
        active_variant_id = self._model_activation_state.active_model_variant_id
        device = self._model_activation_state.device
        needs_reload = (
            self._loaded_model is None
            or self._loaded_model.model_id != active_model_id
            or self._loaded_model.variant_id != active_variant_id
            or self._loaded_model.device != device
        )
        if needs_reload:
            logger.info(
                "Loading model with ID '{}', variant '{}', on device '{}'", active_model_id, active_variant_id, device
            )
            self._unload_model()
            try:
                # Ensure all necessary model files exist before loading the model
                model_xml_path = self._get_model_file_path(
                    project_id=project_id,
                    model_id=active_model_id,
                    variant_id=active_variant_id,
                    extension="xml",
                )
                _ = self._get_model_file_path(
                    project_id=project_id,
                    model_id=active_model_id,
                    variant_id=active_variant_id,
                    extension="bin",
                )
                self._loaded_model = ModelLoader.load(
                    model_id=active_model_id, variant_id=active_variant_id, model_xml_path=model_xml_path, device=device
                )
            except FileNotFoundError:
                logger.exception("Failed to load model with ID '{}'", active_model_id)
                return None
            # The model is created with the threshold embedded in its files; override it if the pipeline says so.
            self._apply_confidence_threshold()

        return self._loaded_model

    def refresh_inference_params(self) -> None:
        """Re-read the pipeline inference parameters and apply them to the loaded model, without reloading it."""
        with get_db_session() as db:
            confidence_threshold = ActiveModelRepo(db=db).get_active_confidence_threshold()
            project_id = self._model_activation_state.project_id
            if project_id is not None:
                # Labels may have been recoloured/renamed in the meantime; keep the overlay in sync.
                self._model_activation_state.label_colors = self._load_label_colors(db=db, project_id=str(project_id))
        self._model_activation_state.confidence_threshold = confidence_threshold
        self._apply_confidence_threshold()

    def _apply_confidence_threshold(self) -> None:
        """Push the configured confidence threshold onto the loaded model."""
        confidence_threshold = self._model_activation_state.confidence_threshold
        if self._loaded_model is None or confidence_threshold is None:
            return
        logger.info(
            "Setting confidence threshold of model '{}' to {}", self._loaded_model.model_id, confidence_threshold
        )
        self._loaded_model.model.set_param("confidence_threshold", confidence_threshold)

    def _unload_model(self) -> None:
        """Release the currently loaded model and free its resources."""
        if self._loaded_model is not None:
            logger.debug("Unloading model '{}'", self._loaded_model.model_id)
            ModelLoader.unload(self._loaded_model)
            self._loaded_model = None
