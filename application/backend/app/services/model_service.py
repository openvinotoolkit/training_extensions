# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import polars as pl
from defusedxml.ElementTree import parse as parse_xml
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import EvaluationDB, MetricScoreDB, ModelRevisionDB, ModelVariantDB
from app.models import EvaluationResult, ModelRevision, ModelVariant, TrainingStatus
from app.models.model_revision import ModelFormat, ModelPrecision
from app.models.system import DeviceInfo
from app.models.training_configuration.configuration import TrainingConfiguration
from app.repositories import (
    EvaluationRepository,
    LabelRepository,
    ModelRevisionRepository,
    ModelVariantRepository,
    PipelineRepository,
)
from app.services.dataset_revision_service import DatasetRevisionService
from app.services.model_manifest_service import ModelManifestService
from app.utils.onnx_metadata import read_onnx_metadata_attrs

from .base import BaseSessionManagedService, ResourceInUseError, ResourceNotFoundError, ResourceType
from .parent_process_guard import parent_process_only


@dataclass
class MetricDisplayInfo:
    display_name: str
    frequency: str = "step"  # "step" or "epoch"

    @property
    def x_axis_label(self) -> str:
        return self.frequency.capitalize()


KEY_MAPPING = {
    "epoch": MetricDisplayInfo(display_name="Epoch", frequency="epoch"),
    "step": MetricDisplayInfo(display_name="Step", frequency="step"),
    # Epoch based learning rate metrics
    "lr": MetricDisplayInfo(display_name="Learning rate", frequency="epoch"),
    "lr-SGD": MetricDisplayInfo(display_name="Learning rate (SGD)", frequency="epoch"),
    "lr-SGD-1": MetricDisplayInfo(display_name="Learning rate (SGD-1)", frequency="epoch"),
    "lr-SGD-1-momentum": MetricDisplayInfo(display_name="Learning rate momentum (SGD-1)", frequency="epoch"),
    "lr-SGD-momentum": MetricDisplayInfo(display_name="Learning rate momentum (SGD)", frequency="epoch"),
    # Step based training metric
    "train/data_time": MetricDisplayInfo(display_name="Training data time", frequency="step"),
    "train/iter_time": MetricDisplayInfo(display_name="Training iteration time", frequency="step"),
    # "train/loss": MetricDisplayInfo(display_name="Training loss", frequency="step"),  # see issue #6350
    "train/loss_bbox": MetricDisplayInfo(display_name="Training loss bbox", frequency="step"),
    "train/loss_dfl": MetricDisplayInfo(display_name="Training loss DFL", frequency="epoch"),
    "train/loss_centerness": MetricDisplayInfo(display_name="Training loss centerness", frequency="step"),
    "train/loss_cls": MetricDisplayInfo(display_name="Training loss cls", frequency="step"),
    "train/loss_mask": MetricDisplayInfo(display_name="Training loss mask", frequency="step"),
    "train/loss_obj": MetricDisplayInfo(display_name="Training loss obj", frequency="step"),
    "train/total_loss": MetricDisplayInfo(display_name="Training total loss", frequency="step"),
    # Epoch based validation metrics
    "val/accuracy": MetricDisplayInfo(display_name="Validation accuracy", frequency="epoch"),
    "val/classes": MetricDisplayInfo(display_name="Validation classes", frequency="epoch"),
    "val/f1-score": MetricDisplayInfo(display_name="Validation F1 score", frequency="epoch"),
    "val/map": MetricDisplayInfo(display_name="Validation mAP", frequency="epoch"),
    "val/map_50": MetricDisplayInfo(display_name="Validation mAP@50", frequency="epoch"),
    "val/map_75": MetricDisplayInfo(display_name="Validation mAP@75", frequency="epoch"),
    "val/map_large": MetricDisplayInfo(display_name="Validation mAP large", frequency="epoch"),
    "val/map_medium": MetricDisplayInfo(display_name="Validation mAP medium", frequency="epoch"),
    "val/map_per_class": MetricDisplayInfo(display_name="Validation mAP per class", frequency="epoch"),
    "val/map_small": MetricDisplayInfo(display_name="Validation mAP small", frequency="epoch"),
    "val/mar_1": MetricDisplayInfo(display_name="Validation mAR@1", frequency="epoch"),
    "val/mar_10": MetricDisplayInfo(display_name="Validation mAR@10", frequency="epoch"),
    "val/mar_100": MetricDisplayInfo(display_name="Validation mAR@100", frequency="epoch"),
    "val/mar_100_per_class": MetricDisplayInfo(display_name="Validation mAR@100 per class", frequency="epoch"),
    "val/mar_large": MetricDisplayInfo(display_name="Validation mAR large", frequency="epoch"),
    "val/mar_medium": MetricDisplayInfo(display_name="Validation mAR medium", frequency="epoch"),
    "val/mar_small": MetricDisplayInfo(display_name="Validation mAR small", frequency="epoch"),
    "val/precision": MetricDisplayInfo(display_name="Validation precision", frequency="epoch"),
    "val/recall": MetricDisplayInfo(display_name="Validation recall", frequency="epoch"),
    "validation/data_time": MetricDisplayInfo(display_name="Validation data time", frequency="epoch"),
    "validation/iter_time": MetricDisplayInfo(display_name="Validation iteration time", frequency="epoch"),
}

# Key under which the confidence threshold is stored in the metadata of exported models
CONFIDENCE_THRESHOLD_IR_PATH = "rt_info/model_info/confidence_threshold"
CONFIDENCE_THRESHOLD_ONNX_KEY = "model_info confidence_threshold"


def _read_ir_confidence_threshold(model_xml: Path) -> float | None:
    """Read the confidence threshold from the ``rt_info`` section of an OpenVINO IR XML file."""
    # The IR topology is parsed directly rather than through openvino.Core.read_model, which would also
    # load the (much larger) weights file.
    root = parse_xml(model_xml).getroot()
    element = root.find(CONFIDENCE_THRESHOLD_IR_PATH) if root is not None else None
    if element is None:
        return None
    value = element.get("value")
    return float(value) if value is not None else None


def _read_onnx_confidence_threshold(model_onnx: Path) -> float | None:
    """Read the confidence threshold from the metadata properties of an ONNX model file."""
    attrs = read_onnx_metadata_attrs(model_onnx, keys=[CONFIDENCE_THRESHOLD_ONNX_KEY])
    value = attrs.get(CONFIDENCE_THRESHOLD_ONNX_KEY)
    return float(value) if value is not None else None


@lru_cache
def _cached_optimal_confidence_threshold(variant_dir: Path, model_format: ModelFormat) -> float | None:
    """
    Read the optimal confidence threshold embedded in an exported model file.

    The value is baked into the model at export time and never changes, so it is cached.

    Args:
        variant_dir (Path): Directory holding the files of the model variant.
        model_format (ModelFormat): Format of the model variant.

    Returns:
        float | None: The optimal confidence threshold, or None if the format does not embed one,
            the model file is missing, or the value could not be read.
    """
    match model_format:
        case ModelFormat.OPENVINO:
            model_file, reader = variant_dir / "model.xml", _read_ir_confidence_threshold
        case ModelFormat.ONNX:
            model_file, reader = variant_dir / "model.onnx", _read_onnx_confidence_threshold
        case _:
            return None

    if not model_file.exists():
        return None
    try:
        return reader(model_file)
    except Exception as exc:
        logger.warning("Could not read the optimal confidence threshold from '{}': {}", model_file, exc)
        return None


@dataclass(frozen=True)
class ModelRevisionMetadata:
    model_id: UUID
    model_name: str
    project_id: UUID
    architecture_id: str
    parent_revision_id: UUID | None
    dataset_revision_id: UUID | None
    training_status: TrainingStatus
    training_configuration: TrainingConfiguration | None = None


class ModelService(BaseSessionManagedService):
    """Service to register and activate models"""

    def __init__(self, data_dir: Path, db_session: Session | None = None) -> None:
        super().__init__(db_session)
        self._projects_dir = data_dir / "projects"

    def get_model(self, project_id: UUID, model_id: UUID) -> ModelRevision:
        """
        Get a model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve.

        Returns:
            ModelRevision: The model revision object containing the model's information.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if not model_rev_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        return ModelRevision.model_validate(model_rev_db)

    def get_model_revision_architecture(self, project_id: UUID, model_id: UUID) -> str:
        """
        Get the architecture ID of a model revision.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.

        Returns:
            str: The architecture ID of the model revision.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if not model_rev_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        return model_rev_db.architecture

    def get_model_license(self, project_id: UUID, model_id: UUID) -> str:
        """
        Get the license of a model by looking up its architecture in the model manifest.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.

        Returns:
            str: The license string (e.g., "Apache 2.0", "AGPL-3.0").

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        architecture = self.get_model_revision_architecture(project_id, model_id)
        manifest = ModelManifestService.get_model_manifest_by_id(architecture)
        return manifest.license

    def get_model_variants(self, project_id: UUID, model_id: UUID) -> list[ModelVariant]:
        """
        Get all variants and their information of a model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve variants for.

        Returns:
            list[ModelVariant]: A list of the model variants.
        """
        model_variant_repo = ModelVariantRepository(db=self.db_session)
        variant_dbs = model_variant_repo.list_by_model_revision(str(model_id))
        variants = []
        for v_db in variant_dbs:
            variant = ModelVariant.model_validate(v_db)
            # Compute weights_size and read the embedded confidence threshold from the filesystem
            variant_dir = self._get_variant_dir(project_id, model_id, UUID(v_db.id))
            if not v_db.files_deleted and variant_dir.exists():
                variant.weights_size = sum(f.stat().st_size for f in variant_dir.iterdir() if f.is_file())
                variant.optimal_confidence_threshold = _cached_optimal_confidence_threshold(variant_dir, variant.format)
            variants.append(variant)
        return variants

    def get_model_size_in_bytes(self, project_id: UUID, model_id: UUID) -> int:
        """
        Get the total size of the model and all its files in bytes.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve size for.
        Returns:
            int: Total size of the model and all its files in bytes.
        """
        model_revision = self.get_model(project_id=project_id, model_id=model_id)
        if model_revision.files_deleted:
            return 0

        model_path = self._projects_dir / str(project_id) / "models" / str(model_id)

        return sum(f.stat().st_size for f in model_path.glob("**/*") if f.is_file())

    def rename_model(self, project_id: UUID, model_id: UUID, model_metadata: dict[str, str]) -> ModelRevision:
        """
        Rename a model revision.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve.
            model_metadata: Dict containing updated model revision name

        Returns:
            ModelRevision: The model revision object containing the model's updated information.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if not model_rev_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        new_name = model_metadata.get("name")
        if new_name is not None:
            model_rev_db.name = new_name
            model_rev_repo.update(model_rev_db)
        return ModelRevision.model_validate(model_rev_db)

    def delete_model(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete a model.

        Deletes a model revision from the database and deletes the folder from the filesystem
        associated with this model. Moreover, if no other models are linked to the linked
        dataset revision, deletes the dataset revision from the database and deletes its
        associated files from the filesystem.

        Args:
            project_id (UUID): The unique identifier of the project whose models to delete.
            model_id (UUID): The unique identifier of the model to delete.

        Returns:
            None

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
            ResourceInUseError: If the model cannot be deleted due to integrity constraints
                (e.g., the model is referenced by other entities).
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_to_delete = model_rev_repo.get_by_id(str(model_id))
        if model_to_delete is None:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        active_pipeline = PipelineRepository(db=self.db_session).get_active_pipeline()
        if active_pipeline and active_pipeline.model_revision_id == str(model_id):
            raise ResourceInUseError(ResourceType.MODEL, str(model_id))

        path = self._projects_dir / str(project_id) / "models" / str(model_id)
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info("Deleted model files at '{}'", path)
            except PermissionError as exc:
                if getattr(exc, "winerror", None) == 32:
                    raise ResourceInUseError(ResourceType.MODEL, str(model_id))
                raise

        try:
            deleted = model_rev_repo.delete(str(model_id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.MODEL, str(model_id))

        if model_to_delete.training_dataset_id is not None:
            model_list = model_rev_repo.list_all(training_dataset_id=model_to_delete.training_dataset_id)
            if len(model_list) == 0:
                dataset_rev_service = DatasetRevisionService(
                    data_dir=self._projects_dir.parent, db_session=self.db_session
                )
                dataset_rev_service.delete_dataset_revision(
                    project_id=UUID(model_to_delete.project_id),
                    revision_id=UUID(model_to_delete.training_dataset_id),
                )

    @parent_process_only
    def delete_model_files(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete only the model files from disk, keeping the model revision record in the database and setting its
        files_deleted flag to True.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
            ResourceInUseError: If the model is currently active in a pipeline.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if model_rev_db is None:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        active_pipeline = PipelineRepository(db=self.db_session).get_active_pipeline()
        if active_pipeline and active_pipeline.model_revision_id == str(model_id):
            raise ResourceInUseError(ResourceType.MODEL, str(model_id))

        model_rev_db.files_deleted = True
        model_rev_repo.update(model_rev_db)

        path = self._projects_dir / str(project_id) / "models" / str(model_id)
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info("Deleted model files at '{}'", path)
            except PermissionError as exc:
                if getattr(exc, "winerror", None) == 32:
                    raise ResourceInUseError(ResourceType.MODEL, str(model_id))
                raise

    def list_models(self, project_id: UUID, dataset_revision_id: UUID | None = None) -> list[ModelRevision]:
        """
        Get information about all available model revisions in a project.

        Retrieves a list of all model revisions that belong to the specified project.
        Each model revision is converted to a schema object containing the model's
        metadata and configuration information.

        Args:
            project_id (UUID): The unique identifier of the project whose models to list.
            dataset_revision_id (UUID | None, optional): The unique identifier of the dataset revision to filter on.

        Returns:
            list[ModelRevision]: A list of model revision objects representing all model
                revisions in the project, optionally filtered on dataset revision.
                Returns an empty list if the project has no models.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        training_dataset_id = str(dataset_revision_id) if dataset_revision_id is not None else None
        return [
            ModelRevision.model_validate(model_rev_db)
            for model_rev_db in model_rev_repo.list_all(training_dataset_id=training_dataset_id)
        ]

    def create_revision(self, metadata: ModelRevisionMetadata) -> None:
        """
        Create and persist a new model revision for the given project metadata.

        Reads the project's label definitions, serializes them into a dict format,
        combines them with the provided metadata into a new model revision record,
        and saves it to the database.

        Args:
            metadata (ModelRevisionMetadata): Metadata used to create the new model revision
                including project id, architecture, optional parent revision id,
                dataset revision id, training status and optional training
                configuration.
        """
        project_id = str(metadata.project_id)
        label_repo = LabelRepository(project_id=project_id, db=self.db_session)
        labels_schema_rev = {"labels": [{"name": label.name, "id": label.id} for label in label_repo.list_all()]}

        model_revision_repo = ModelRevisionRepository(project_id=project_id, db=self.db_session)
        model_revision_repo.save(
            ModelRevisionDB(
                id=str(metadata.model_id),
                name=metadata.model_name,
                project_id=str(metadata.project_id),
                architecture=metadata.architecture_id,
                parent_revision=str(metadata.parent_revision_id) if metadata.parent_revision_id else None,
                training_status=metadata.training_status,
                training_configuration=metadata.training_configuration.model_dump()
                if metadata.training_configuration
                else {},
                training_dataset_id=str(metadata.dataset_revision_id),
                label_schema_revision=labels_schema_rev,
            )
        )

    def update_revision_status(
        self,
        project_id: UUID,
        model_id: UUID,
        training_status: TrainingStatus,
        training_started_at: datetime | None = None,
        training_finished_at: datetime | None = None,
        training_device: DeviceInfo | None = None,
    ) -> None:
        """
        Updates the training status of a model revision for the given project.

        Args:
            project_id (UUID): Identifier of the project that owns the model revision.
            model_id (UUID): Identifier of the model revision to update.
            training_status (TrainingStatus): New training status to set for the model revision.
            training_started_at (datetime): Date and time when the training was started
            training_finished_at (datetime): Date and time when the training was finished
            training_device (DeviceInfo | None): Hardware device used to run the training
        """
        model_revision_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_revision_repo.update_training_status(
            obj_id=str(model_id),
            training_status=training_status,
            training_started_at=training_started_at,
            training_finished_at=training_finished_at,
            training_device=training_device.model_dump(mode="json") if training_device is not None else None,
        )

    def get_model_binary_files(
        self, project_id: UUID, model_id: UUID, model_variant_id: UUID
    ) -> tuple[bool, tuple[Path, ...]]:
        """
        Get the paths to the model binary files.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
            model_variant_id (UUID): The unique identifier of the model variant files to retrieve.

        Returns:
            tuple[bool, tuple[Path, ...]]: A tuple where the first element indicates if the files exist,
                and the second element is a tuple of Paths to the model variant's files.

        Raises:
            ResourceNotFoundError: If the model has been marked as deleted.
        """
        model_revision = self.get_model(project_id=project_id, model_id=model_id)
        if model_revision.files_deleted:
            return False, ()

        # Find the variant matching the requested variant ID
        model_variant_repo = ModelVariantRepository(db=self.db_session)
        variant_dbs = model_variant_repo.list_by_model_revision(str(model_id))
        for v_db in variant_dbs:
            if v_db.id == str(model_variant_id) and not v_db.files_deleted:
                return self._get_variant_binary_files(project_id, model_id, UUID(v_db.id))

        return False, ()

    def _get_variant_binary_files(
        self, project_id: UUID, model_id: UUID, variant_id: UUID
    ) -> tuple[bool, tuple[Path, ...]]:
        """
        Get binary files for a specific variant from the filesystem.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
            variant_id (UUID): The unique identifier of the model variant.

        Returns:
            tuple[bool, tuple[Path, ...]]: A tuple where the first element indicates if the files exist,
                and the second element is a tuple of Paths to the model variant's files.
        """
        variant_dir = self._get_variant_dir(project_id, model_id, variant_id)

        xml_file = variant_dir / "model.xml"
        bin_file = variant_dir / "model.bin"
        onnx_file = variant_dir / "model.onnx"
        pt_file = variant_dir / "model.pt"
        metadata_file = variant_dir / "metadata.yaml"

        if xml_file.exists() and bin_file.exists():
            paths: list[Path] = [xml_file, bin_file]
            if metadata_file.exists():
                paths.append(metadata_file)
            return True, tuple(paths)
        if onnx_file.exists():
            paths = [onnx_file]
            if metadata_file.exists():
                paths.append(metadata_file)
            return True, tuple(paths)
        if pt_file.exists():
            return True, (pt_file,)

        return False, ()

    def _get_variant_dir(self, project_id: UUID, model_id: UUID, variant_id: UUID) -> Path:
        """Get the filesystem path for a model variant directory."""
        return self._projects_dir / str(project_id) / "models" / str(model_id) / "variants" / str(variant_id)

    def create_variant(
        self,
        model_revision_id: UUID,
        format: ModelFormat,
        precision: ModelPrecision,
        quantization_info: dict | None = None,
        model_variant_id: UUID | None = None,
    ) -> ModelVariant:
        """
        Create a new model variant record in the database.

        Args:
            model_revision_id: UUID of the parent model revision.
            format: The format of the model variant.
            precision: The precision of the model variant.
            quantization_info: Optional quantization metadata.
            model_variant_id: Optional UUID for the model variant. If not provided, a new UUID will be generated.

        Returns:
            ModelVariant: The created model variant.
        """
        model_variant_repo = ModelVariantRepository(db=self.db_session)
        variant_db = ModelVariantDB(
            id=str(model_variant_id) if model_variant_id else None,
            model_revision_id=str(model_revision_id),
            format=format.value,
            precision=precision.value,
            quantization_info=quantization_info,
        )
        model_variant_repo.save(variant_db)
        return ModelVariant(
            id=UUID(variant_db.id),
            model_revision_id=model_revision_id,
            format=format,
            precision=precision,
            quantization_info=quantization_info,
            evaluations=[],
        )

    def get_variant(self, variant_id: UUID) -> ModelVariant:
        """
        Get a model variant by its ID.

        Args:
            variant_id: The unique identifier of the variant.

        Returns:
            ModelVariant: The model variant.

        Raises:
            ResourceNotFoundError: If the variant is not found.
        """
        model_variant_repo = ModelVariantRepository(db=self.db_session)
        variant_db = model_variant_repo.get_by_id(str(variant_id))
        if not variant_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(variant_id))
        return ModelVariant.model_validate(variant_db)

    def get_optimal_confidence_threshold(self, project_id: UUID, model_id: UUID, variant_id: UUID) -> float | None:
        """
        Get the confidence threshold that was determined for a model variant when it was exported.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
            variant_id (UUID): The unique identifier of the model variant.

        Returns:
            float | None: The optimal confidence threshold, or None if the variant does not embed one
                (e.g. PyTorch variants, or tasks that do not use a confidence threshold).
        """
        variant = self.get_variant(variant_id=variant_id)
        variant_dir = self._get_variant_dir(project_id, model_id, variant_id)
        if variant.files_deleted or not variant_dir.exists():
            return None
        return _cached_optimal_confidence_threshold(variant_dir, variant.format)

    def save_evaluation_result(self, result: EvaluationResult) -> None:
        evaluation_db = EvaluationDB(
            model_revision_id=str(result.model_revision_id),
            model_variant_id=str(result.model_variant_id),
            dataset_revision_id=str(result.dataset_revision_id),
            subset=result.subset,
        )
        metrics_db = [MetricScoreDB(metric=item[0], score=item[1]) for item in result.metrics.items()]
        evaluation_db.metric_scores = metrics_db

        evaluation_repo = EvaluationRepository(self.db_session)
        evaluation_repo.save(evaluation_db)

    def get_model_training_metrics(
        self,
        project_id: UUID,
        model_id: UUID,
    ) -> list[dict]:
        """
        Get training metrics from the metrics.csv file for a model.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.

        Returns:
            list[dict]: A list of metric dictionaries

        Raises:
            ResourceNotFoundError: If the model or metrics file is not found.
        """
        # Verify the model exists
        model_revision = self.get_model(project_id=project_id, model_id=model_id)
        if model_revision.files_deleted:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        # training + validation metrics are version_0, testing metrics are version_1
        metrics_file = (
            self._projects_dir / str(project_id) / "models" / str(model_id) / "metrics" / "version_0" / "metrics.csv"
        )
        if not metrics_file.exists():
            raise ResourceNotFoundError(ResourceType.MODEL, f"{model_id} (metrics.csv not found)")

        return self._parse_metrics_csv(metrics_file)

    @staticmethod
    def _parse_metrics_csv(metrics_file: Path) -> list[dict]:
        """
        Parse a metrics CSV file and return the formatted metrics.

        For each metric column (excluding 'epoch' and 'step'):
        1. Filter out rows with null values in the metric column
        2. Build a TrainingMetrics object for that metric

        Args:
            metrics_file (Path): Path to the metrics.csv file.

        Returns:
            list[dict]: A list of formatted metric dictionaries.
        """
        metrics: list[dict] = []

        # Read the CSV file with polars
        df = pl.read_csv(metrics_file)

        # Due to a quirk in SimpleLearningRateMonitor/LearningRateMonitor, the LR metric does not log the epoch value.
        # Fill in missing epoch values.
        df = df.with_columns(
            pl.when(pl.col("epoch").is_null() & pl.col("epoch").shift(-1).is_not_null())
            .then(pl.col("epoch").shift(-1))
            .otherwise(pl.col("epoch"))
            .alias("epoch")
        )

        for col in df.columns:
            if col in ["epoch", "step"]:
                continue

            mapped_metric = KEY_MAPPING.get(col)
            if mapped_metric is None:
                logger.debug("Metric '{}' is not in KEY_MAPPING, skipping", col)
                continue

            display_name = mapped_metric.display_name
            frequency = mapped_metric.frequency
            x_axis_label = mapped_metric.x_axis_label

            # Filter to include only 'epoch', 'step', and the current metric column
            # Then filter out rows where the metric column has null values
            metric_df = df.select(["epoch", "step", col]).filter(pl.col(col).is_not_null())

            if metric_df.is_empty():
                logger.debug("Metric '{}' has no non-null values, skipping", col)
                continue

            points = [
                {"x": float(row[frequency]), "y": float(row[col]), "type": "point"}
                for row in metric_df.iter_rows(named=True)
            ]

            metric = {
                "header": display_name,
                "type": "line",
                "key": display_name,
                "value": {
                    "x_axis_label": x_axis_label,
                    "y_axis_label": display_name,
                    "line_data": [
                        {
                            "header": display_name,
                            "key": display_name,
                            "points": points,
                        }
                    ],
                },
            }
            metrics.append(metric)

        return metrics

    def get_logs(self, project_id: UUID, model_id: UUID, as_text: bool = False) -> Path | Iterator[str] | None:
        """
        Get the training logs for a model revision.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
            as_text (bool): If True, parse NDJSON and return plain text. If False, return file path.

        Returns:
            Path | Iterator[str] | None: Path to the training log file (if as_text=False),
                iterator yielding plain text log lines (if as_text=True), or None if logs don't exist.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
            ValueError: If the model is in NOT_STARTED or IN_PROGRESS status.
        """
        model_revision = self.get_model(project_id=project_id, model_id=model_id)

        if model_revision.training_info and model_revision.training_info.status in (
            TrainingStatus.NOT_STARTED,
            TrainingStatus.IN_PROGRESS,
        ):
            raise ValueError(
                "Logs are not available for models that have not started or are currently in progress of training"
            )

        log_file = self._projects_dir / str(project_id) / "models" / str(model_id) / "training.log"

        if not log_file.exists():
            return None

        if not as_text:
            return log_file

        def _iter_text_lines() -> Iterator[str]:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        text = entry.get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse log line: {}", e)
                        yield "[MALFORMED LOG LINE]\n"

        return _iter_text_lines()
