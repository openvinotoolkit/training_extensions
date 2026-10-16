# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from getitune.backend.openvino.engine import OVEngine
    from getitune.data.module import DataModule

import yaml
from datumaro.experimental.fields import Subset
from loguru import logger
from sqlalchemy.orm import Session

from app.execution.base import Execution, step
from app.execution.common.geti_config_converter import GetiConfigConverter
from app.execution.common.getitune_converters import (
    build_getitune_subset_config,
    convert_metrics,
    get_getitune_dataset_class_by_task_type,
    get_getitune_task_type_by_task,
    get_metric_by_task,
)
from app.models import DatasetItemSubset, EvaluationResult, ModelRevision, TrainingStatus
from app.models.jobs.quantization_job import QuantizationJobParams
from app.models.model_revision import ModelFormat, ModelPrecision, ModelVariant
from app.models.project import Task
from app.services import DatasetRevisionService, ModelService, ProjectService, TrainingConfigurationService


@dataclass(frozen=True)
class QuantizationDependencies:
    data_dir: Path
    model_service: ModelService
    dataset_revision_service: DatasetRevisionService
    project_service: ProjectService
    training_configuration_service: TrainingConfigurationService
    db_session_factory: Callable[[], AbstractContextManager[Session]]


class GetiTuneQuantizer(Execution[QuantizationJobParams]):
    """getitune-specific quantization implementation using OVEngine."""

    params_type = QuantizationJobParams

    def __init__(self, quantization_deps: QuantizationDependencies) -> None:
        super().__init__()
        self._data_dir = quantization_deps.data_dir
        self._model_service = quantization_deps.model_service
        self._dataset_revision_service = quantization_deps.dataset_revision_service
        self._project_service = quantization_deps.project_service
        self._training_configuration_service = quantization_deps.training_configuration_service
        self._db_session_factory = quantization_deps.db_session_factory

    @step("Validate Model", 5)
    def validate_model(self, params: QuantizationJobParams) -> tuple[ModelRevision, ModelVariant]:
        """Verify the source model is valid for quantization.

        Returns:
            Tuple of (model, openvino_fp16_variant).
        """
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            model = self._model_service.get_model(project_id=params.project_id, model_id=params.model_id)

        # Check training completed successfully
        if model.training_info is None or model.training_info.status != TrainingStatus.SUCCESSFUL:
            raise ValueError(
                f"Model {params.model_id} has not completed training successfully. "
                f"Current status: {model.training_info.status if model.training_info else 'unknown'}"
            )

        # Check files are not deleted
        if model.files_deleted:
            raise FileNotFoundError(f"Model {params.model_id} files have been deleted")

        # Check OpenVINO IR files exist
        openvino_variant = self._get_openvino_fp16_variant(model)
        if openvino_variant is None:
            raise FileNotFoundError(f"Model {params.model_id} does not have an OpenVINO FP16 variant")

        ov_model_xml_path = self._get_variant_model_xml_path(params.project_id, params.model_id, openvino_variant.id)
        if not ov_model_xml_path.exists():
            raise FileNotFoundError(f"OpenVINO model files not found at {ov_model_xml_path}")

        # Validate metadata.yaml structure if it exists (optional — Lightning models don't produce it)
        metadata_path = ov_model_xml_path.parent / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = yaml.safe_load(f)
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata.yaml is malformed (not a dict) at {metadata_path}")
            if "args" not in metadata or not isinstance(metadata["args"], dict):
                raise ValueError(f"metadata.yaml is missing 'args' dict at {metadata_path}")

        logger.info("Model {} validated for quantization (architecture={})", model.id, model.architecture)
        return model, openvino_variant

    @step("Prepare Calibration Dataset", 20)
    def prepare_calibration_dataset(
        self,
        params: QuantizationJobParams,
        model: ModelRevision,
    ) -> DataModule:
        """Load and prepare the calibration dataset from the training dataset revision."""
        from getitune.data.entity.utils import detect_storage_dtype
        from getitune.data.module import DataModule

        # Get the dataset revision used for training
        dataset_revision_id = model.training_info.dataset_revision_id if model.training_info else None
        if dataset_revision_id is None:
            raise ValueError(f"Model {params.model_id} does not have an associated dataset revision")

        # Load training configuration to get data pipeline settings
        with self._db_session_factory() as db:
            self._training_configuration_service.set_db_session(db)
            training_config = self._training_configuration_service.get_by_model_revision(
                project_id=params.project_id,
                model_revision_id=params.model_id,
            )

        # Convert to getitune config format
        with self._db_session_factory() as db:
            self._project_service.set_db_session(db)
            project = self._project_service.get_project_by_id(params.project_id)
        task = project.task

        geti_training_config = training_config.model_dump(exclude_none=True)
        geti_training_config["hyper_parameters"] = geti_training_config.pop("algo_level_parameters")
        geti_training_config["model_manifest_id"] = model.architecture
        geti_training_config["sub_task_type"] = get_getitune_task_type_by_task(task)

        converter = GetiConfigConverter()
        getitune_training_config = converter.convert(geti_training_config)

        # Load the dataset revision
        with self._db_session_factory() as db:
            self._dataset_revision_service.set_db_session(db)
            dm_dataset = self._dataset_revision_service.load_revision(
                project_id=params.project_id,
                dataset_revision_id=dataset_revision_id,
            )

        # Extract subsets
        dm_training_dataset = dm_dataset.filter_by_subset(Subset.TRAINING)
        dm_validation_dataset = dm_dataset.filter_by_subset(Subset.VALIDATION)
        dm_testing_dataset = dm_dataset.filter_by_subset(Subset.TESTING)

        # Build subset configs
        train_subset_config = build_getitune_subset_config(getitune_training_config, "train")
        val_subset_config = build_getitune_subset_config(getitune_training_config, "val")
        test_subset_config = build_getitune_subset_config(getitune_training_config, "test")

        # Detect storage dtype and propagate to subset configs.
        storage_dtype, num_channels = detect_storage_dtype(dm_training_dataset)
        for cfg in (train_subset_config, val_subset_config, test_subset_config):
            cfg.intensity.storage_dtype = storage_dtype
            if num_channels == 1:
                cfg.intensity.repeat_channels = 3

        # Wrap into VisionDataset instances
        getitune_task_type = get_getitune_task_type_by_task(task)
        getitune_dataset_class = get_getitune_dataset_class_by_task_type(getitune_task_type)

        getitune_training_dataset = getitune_dataset_class(
            dm_subset=dm_training_dataset,
            transforms=train_subset_config.transforms,  # pyrefly: ignore[missing-attribute,bad-argument-type]
        )
        getitune_validation_dataset = getitune_dataset_class(
            dm_subset=dm_validation_dataset,
            transforms=val_subset_config.transforms,  # pyrefly: ignore[missing-attribute,bad-argument-type]
        )
        getitune_testing_dataset = getitune_dataset_class(
            dm_subset=dm_testing_dataset,
            transforms=test_subset_config.transforms,  # pyrefly: ignore[missing-attribute,bad-argument-type]
        )

        # Build the DataModule
        getitune_datamodule = DataModule.from_vision_datasets(
            train_dataset=getitune_training_dataset,
            val_dataset=getitune_validation_dataset,
            test_dataset=getitune_testing_dataset,
            train_subset=train_subset_config,
            val_subset=val_subset_config,
            test_subset=test_subset_config,
        )

        logger.info(
            "Calibration dataset prepared with {} training, {} validation, {} testing samples",
            len(getitune_training_dataset),
            len(getitune_validation_dataset),
            len(getitune_testing_dataset),
        )
        return getitune_datamodule

    @step("Initialize OV Engine", 25)
    def initialize_engine(
        self,
        params: QuantizationJobParams,
        model: ModelRevision,
        datamodule: DataModule,
    ) -> OVEngine:
        from getitune.backend.openvino.engine import OVEngine

        """Create the OVEngine for quantization."""
        openvino_variant = self._get_openvino_fp16_variant(model)
        if openvino_variant is None:
            raise FileNotFoundError(f"Model {params.model_id} does not have an OpenVINO FP16 variant")
        ov_model_xml_path = self._get_variant_model_xml_path(params.project_id, params.model_id, openvino_variant.id)

        logger.info("Initializing OVEngine with model at {}", ov_model_xml_path)
        return OVEngine(
            model=ov_model_xml_path,
            data=datamodule,
            work_dir=self._data_dir / f"getitune-quantize-workspace-{params.model_id}",
        )

    @step("Run Quantization", 80)
    def run_quantization(
        self,
        ov_engine: OVEngine,
        subset_size: int,
        max_drop: float | None = None,
        max_num_iterations: int | None = None,
    ) -> Path:
        """Execute the quantization process using nncf.quantize() via OVEngine.optimize().

        Args:
            ov_engine: The OVEngine instance.
            subset_size: Maximum calibration subset size.
            max_drop: Optional maximum accuracy drop for accuracy-aware quantization.
            max_num_iterations: Optional maximum number of iterations for accuracy-aware
                quantization. ``None`` means unlimited. Only used when ``max_drop`` is set.

        Returns:
            Path to the quantized model XML file.
        """
        logger.info("Running quantization with max_calibration_subset_size={}", subset_size)
        quantized_model_path = ov_engine.optimize(
            max_data_subset_size=subset_size,
            max_drop=max_drop,
            max_num_iterations=max_num_iterations,
        )
        logger.info("Quantization completed. Model saved at {}", quantized_model_path)
        return quantized_model_path

    @step("Evaluate Quantized Model", 95)
    def evaluate_quantized_model(
        self,
        ov_engine: OVEngine,
        quantized_model_path: Path,
        task: Task,
        model_revision_id: UUID,
        model_variant_id: UUID,
        dataset_revision_id: UUID,
    ) -> None:
        """Evaluate the quantized model on the testing subset."""
        logger.info("Evaluating the quantized model on the testing set...")
        metrics = ov_engine.test(checkpoint=quantized_model_path, metric=get_metric_by_task(task))
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            self._model_service.save_evaluation_result(
                EvaluationResult(
                    model_revision_id=model_revision_id,
                    model_variant_id=model_variant_id,
                    dataset_revision_id=dataset_revision_id,
                    subset=DatasetItemSubset.TESTING,
                    metrics=convert_metrics(metrics),
                )
            )

    @step("Store Artifacts", 100)
    def store_artifacts(
        self,
        params: QuantizationJobParams,
        quantized_model_path: Path,
        model_variant_id: UUID,
        source_variant_id: UUID,
    ) -> None:
        """Store quantized model files.

        The getitune workspace itself is removed by ``QuantizationJob.on_complete``
        after the job terminates, so this step does not clean it up.
        """
        model_dir = self._base_model_path(params.project_id, params.model_id)
        variant_dir = model_dir / "variants" / str(model_variant_id)
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Copy quantized model files (XML + BIN)
        shutil.copyfile(quantized_model_path, variant_dir / "model.xml")
        bin_path = quantized_model_path.with_suffix(".bin")
        if bin_path.exists():
            shutil.copyfile(bin_path, variant_dir / "model.bin")

        # Copy metadata.yaml from the source FP16 variant and update for INT8
        source_variant_dir = model_dir / "variants" / str(source_variant_id)
        source_metadata_path = source_variant_dir / "metadata.yaml"
        if source_metadata_path.exists():
            with open(source_metadata_path) as f:
                metadata = yaml.safe_load(f)
            metadata["args"]["int8"] = True
            metadata["args"]["half"] = False
            dest_metadata_path = variant_dir / "metadata.yaml"
            with open(dest_metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
            logger.info("Wrote INT8 metadata.yaml to {}", dest_metadata_path)

        logger.info("Stored quantized model variant at {}", variant_dir)

    def execute(self, params: QuantizationJobParams) -> None:
        """Execute the full quantization pipeline."""
        model, openvino_fp16_variant = self.validate_model(params=params)

        with self._db_session_factory() as db:
            self._project_service.set_db_session(db)
            project = self._project_service.get_project_by_id(params.project_id)
        task = project.task

        datamodule = self.prepare_calibration_dataset(params=params, model=model)
        ov_engine = self.initialize_engine(params=params, model=model, datamodule=datamodule)

        # max_num_iterations only applies to accuracy-aware quantization (max_drop set).
        # For standard PTQ it must always be None regardless of the configured value.
        max_num_iterations = params.max_num_iterations if params.max_drop is not None else None

        quantized_model_path = self.run_quantization(
            ov_engine=ov_engine,
            subset_size=params.max_calibration_subset_size,
            max_drop=params.max_drop,
            max_num_iterations=max_num_iterations,
        )

        quantization_info = {
            "type": "PTQ" if params.max_drop is None else "Accuracy-aware PTQ",
            "max_calibration_subset_size": params.max_calibration_subset_size,
            "max_drop": params.max_drop,
            "max_num_iterations": max_num_iterations,
        }

        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            variant = self._model_service.create_variant(
                model_revision_id=params.model_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.INT8,
                quantization_info=quantization_info,
                model_variant_id=params.model_variant_id,
            )
        logger.info("Created INT8 variant record (id={})", variant.id)

        self.evaluate_quantized_model(
            ov_engine=ov_engine,
            quantized_model_path=quantized_model_path,
            task=task,
            model_revision_id=params.model_id,
            model_variant_id=variant.id,
            dataset_revision_id=model.training_info.dataset_revision_id,
        )

        self.store_artifacts(
            params=params,
            quantized_model_path=quantized_model_path,
            model_variant_id=variant.id,
            source_variant_id=openvino_fp16_variant.id,
        )

    def _base_model_path(self, project_id: UUID, model_id: UUID) -> Path:
        return self._data_dir / "projects" / str(project_id) / "models" / str(model_id)

    def _get_variant_model_xml_path(self, project_id: UUID, model_id: UUID, variant_id: UUID) -> Path:
        """Get the path to the OpenVINO model XML file for a given variant."""
        return self._base_model_path(project_id, model_id) / "variants" / str(variant_id) / "model.xml"

    @staticmethod
    def _get_openvino_fp16_variant(model: ModelRevision) -> ModelVariant | None:
        """Find the OpenVINO FP16 variant of a model."""
        for variant in model.variants:
            if (
                variant.format == ModelFormat.OPENVINO
                and variant.precision == ModelPrecision.FP16
                and not variant.files_deleted
            ):
                return variant
        return None

    @staticmethod
    def _get_int8_variant(model: ModelRevision) -> ModelVariant | None:
        """Find an existing INT8 variant of a model."""
        for variant in model.variants:
            if variant.precision == ModelPrecision.INT8 and not variant.files_deleted:
                return variant
        return None
