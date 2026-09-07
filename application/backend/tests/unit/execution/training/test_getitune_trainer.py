# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch
from uuid import uuid4

import numpy as np
import pytest
import torch
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import ImageInfo, Subset
from getitune import TaskType as GetiTuneTaskType
from getitune.config.data import IntensityConfig
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable
from getitune.metrics.types import MetricCallable
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision

from app.core.jobs.exec.exceptions import CancelledExc
from app.core.run import ExecutionContext
from app.datumaro_converter import SampleMode
from app.datumaro_converter.domain.samples.training import (
    DetectionTrainingSample,
    InstanceSegmentationTrainingSample,
    MulticlassClassificationTrainingSample,
    MultilabelClassificationTrainingSample,
)
from app.execution.base import ExecutionErr
from app.execution.common.geti_config_converter import GetiConfigConverter
from app.execution.training.getitune_trainer import (
    DatasetInfo,
    ExportedModels,
    GetiTuneTrainer,
    ModelVariantDescriptor,
    TrainingDependencies,
)
from app.models import (
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    ModelVariant,
    Task,
    TaskType,
    TrainingJobParams,
    TrainingStatus,
)
from app.models.model_revision import ModelFormat, ModelPrecision
from app.models.system import DeviceInfo, DeviceType
from app.models.training_configuration import AlgoLevelParameters, TaskLevelParameters, TrainingConfiguration
from app.services import ModelRevisionMetadata, ModelService, TrainingConfigurationService
from app.services.base_weights_service import BaseWeightsService
from app.services.subset_assignment import DatasetItemWithLabels, SubsetAssigner, SubsetAssignment, SubsetService


@pytest.fixture
def fxt_weights_service() -> Mock:
    """Create a mock BaseWeightsService."""
    return Mock(spec=BaseWeightsService)


@pytest.fixture
def fxt_subset_service() -> Mock:
    """Mock SubsetService for testing."""
    return Mock(spec=SubsetService)


@pytest.fixture
def fxt_assigner() -> Mock:
    """Mock SubsetAssigner for testing."""
    return Mock(spec=SubsetAssigner)


@pytest.fixture
def fxt_model_service() -> Mock:
    """Mock ModelService for testing."""
    return Mock(spec=ModelService)


@pytest.fixture
def fxt_training_configuration_service() -> Mock:
    """Mock TrainingConfigurationService for testing."""
    return Mock(spec=TrainingConfigurationService)


@pytest.fixture
def fxt_getitune_trainer(
    tmp_path: Path,
    fxt_weights_service: Mock,
    fxt_subset_service: Mock,
    fxt_assigner: Mock,
    fxt_dataset_service: Mock,
    fxt_dataset_revision_service: Mock,
    fxt_model_service: Mock,
    fxt_training_configuration_service: Mock,
    fxt_db_session_factory: Callable,
) -> Callable[[], GetiTuneTrainer]:
    """Create an GetiTuneTrainer instance."""

    def create_getitune_trainer() -> GetiTuneTrainer:
        getitune_trainer = GetiTuneTrainer(
            TrainingDependencies(
                data_dir=tmp_path,
                base_weights_service=fxt_weights_service,
                subset_service=fxt_subset_service,
                subset_assigner=fxt_assigner,
                dataset_service=fxt_dataset_service,
                dataset_revision_service=fxt_dataset_revision_service,
                model_service=fxt_model_service,
                training_configuration_service=fxt_training_configuration_service,
                db_session_factory=fxt_db_session_factory,
            )
        )
        execution_ctx = Mock(spec=ExecutionContext)
        execution_ctx.report = Mock()
        execution_ctx.heartbeat = Mock()
        getitune_trainer._ctx = execution_ctx
        return getitune_trainer

    return create_getitune_trainer


class TestGetiTuneTrainerPrepareWeights:
    """Tests for the GetiTuneTrainer.prepare_weights method."""

    def test_prepare_weights_without_parent_model(
        self,
        fxt_weights_service: Mock,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
    ):
        """Test preparing weights when no parent model revision ID is provided."""
        # Arrange
        training_params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=None,
            job_id=uuid4(),
            project_id=uuid4(),
        )
        getitune_trainer = fxt_getitune_trainer()

        expected_weights_path = Path("/path/to/weights.pth")
        fxt_weights_service.get_local_weights_path.return_value = expected_weights_path

        # Act
        weights_path = getitune_trainer.prepare_weights(training_params)

        # Assert
        assert weights_path == expected_weights_path
        fxt_weights_service.get_local_weights_path.assert_called_once_with(
            task=TaskType.DETECTION, model_manifest_id="object-detection-yolox-s"
        )

    def test_prepare_weights_with_parent_model(
        self,
        tmp_path: Path,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
    ):
        """Test preparing weights when parent model revision ID is provided."""
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        parent_model_variant_id = uuid4()
        training_params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            project_id=project_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=parent_model_revision_id,
            job_id=uuid4(),
        )
        expected_weights_path = (
            tmp_path
            / "projects"
            / str(project_id)
            / "models"
            / str(parent_model_revision_id)
            / "variants"
            / str(parent_model_variant_id)
            / "model.pt"
        )
        expected_weights_path.parent.mkdir(parents=True, exist_ok=True)
        expected_weights_path.touch()
        getitune_trainer = fxt_getitune_trainer()

        getitune_trainer._model_service.get_model_variants.return_value = [  # pyrefly: ignore[missing-attribute]
            ModelVariant(
                id=parent_model_variant_id,
                model_revision_id=parent_model_revision_id,
                format=ModelFormat.PYTORCH,
                precision=ModelPrecision.FP32,
            )
        ]

        # Act
        weights_path = getitune_trainer.prepare_weights(training_params)

        # Assert
        assert weights_path == expected_weights_path

    def test_prepare_weights_with_parent_model_no_variants(
        self,
        tmp_path: Path,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
    ):
        """Test preparing weights when no parent model revision variants are available."""
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        training_params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            project_id=project_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=parent_model_revision_id,
            job_id=uuid4(),
        )
        getitune_trainer = fxt_getitune_trainer()

        getitune_trainer._model_service.get_model_variants.return_value = []  # pyrefly: ignore[missing-attribute]

        # Act
        msg = (
            "Can't start training - the parent revision has no variants (it may have failed). "
            "Review the previous revision and retry."
        )
        with pytest.raises(ExecutionErr, match=re.escape(msg)):
            getitune_trainer.prepare_weights(training_params)

    def test_prepare_weights_with_parent_model_no_file_raises_error(
        self,
        tmp_path: Path,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
    ):
        """Test that FileNotFoundError is raised when parent model weights file is missing."""
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        parent_model_variant_id = uuid4()
        training_params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            project_id=project_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=parent_model_revision_id,
            job_id=uuid4(),
        )
        expected_weights_path = (
            tmp_path
            / "projects"
            / str(project_id)
            / "models"
            / str(parent_model_revision_id)
            / "variants"
            / str(parent_model_variant_id)
            / "model.pt"
        )
        getitune_trainer = fxt_getitune_trainer()

        getitune_trainer._model_service.get_model_variants.return_value = [  # pyrefly: ignore[missing-attribute]
            ModelVariant(
                id=parent_model_variant_id,
                model_revision_id=parent_model_revision_id,
                format=ModelFormat.PYTORCH,
                precision=ModelPrecision.FP32,
            )
        ]

        # Act
        with pytest.raises(FileNotFoundError) as excinfo:
            getitune_trainer.prepare_weights(training_params)
        assert excinfo.value.args[0] == f"Parent model weights not found at {expected_weights_path}"


class TestGetiTuneTrainerPrepareTrainingConfiguration:
    """Tests for the GetiTuneTrainer.prepare_training_configuration method."""

    def test_prepare_training_configuration(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        training_params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            project_id=project_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=parent_model_revision_id,
            job_id=uuid4(),
        )
        getitune_trainer = fxt_getitune_trainer()
        mock_training_config = TrainingConfiguration(
            task_level_parameters=TaskLevelParameters(),
            algo_level_parameters=MagicMock(spec=AlgoLevelParameters),
        )
        getitune_trainer._training_configuration_service.get_by_model_architecture.return_value = mock_training_config  # type: ignore[attr-defined]
        mock_getitune_training_config = {"key2": "value2"}
        with patch.object(GetiConfigConverter, "convert", return_value=mock_getitune_training_config) as mock_convert:
            # Act
            training_config, getitune_training_config = getitune_trainer.prepare_training_configuration(
                training_params=training_params, task=Task(task_type=TaskType.DETECTION)
            )

        # Assert
        getitune_trainer._training_configuration_service.get_by_model_architecture.assert_called_once_with(  # type: ignore[attr-defined]
            project_id=training_params.project_id,
            model_architecture_id=training_params.model_architecture_id,
        )
        expected_getitune_training_config = mock_training_config.model_dump(exclude_none=True)
        expected_getitune_training_config["hyper_parameters"] = expected_getitune_training_config.pop(
            "algo_level_parameters"
        )
        expected_getitune_training_config["model_manifest_id"] = training_params.model_architecture_id
        expected_getitune_training_config["sub_task_type"] = GetiTuneTaskType.DETECTION
        mock_convert.assert_called_once_with(expected_getitune_training_config)
        assert training_config == mock_training_config
        assert getitune_training_config == mock_getitune_training_config


class TestGetiTuneTrainerAssignSubsets:
    """Tests for the GetiTuneTrainer.assign_subsets method."""

    def test_assign_subsets_with_unassigned_items(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_subset_service: Mock,
        fxt_assigner: Mock,
    ):
        """Test assigning subsets when unassigned items exist."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        project_id = uuid4()
        label_1, label_2, label_3 = uuid4(), uuid4(), uuid4()

        # Create mock unassigned items
        unassigned_items = [
            DatasetItemWithLabels(item_id=uuid4(), labels={label_1, label_2}),
            DatasetItemWithLabels(item_id=uuid4(), labels={label_1}),
            DatasetItemWithLabels(item_id=uuid4(), labels={label_1, label_2, label_3}),
        ]
        fxt_subset_service.get_unassigned_items_with_labels.return_value = unassigned_items
        fxt_subset_service.has_all_subsets_assigned.return_value = False

        # Mock configuration
        training_config = TrainingConfiguration(
            task_level_parameters=TaskLevelParameters(),
            algo_level_parameters=MagicMock(spec=AlgoLevelParameters),
        )

        # Mock assignments
        expected_assignments = [
            SubsetAssignment(item_id=unassigned_items[0].item_id, subset=DatasetItemSubset.TRAINING),
            SubsetAssignment(item_id=unassigned_items[1].item_id, subset=DatasetItemSubset.VALIDATION),
            SubsetAssignment(item_id=unassigned_items[2].item_id, subset=DatasetItemSubset.TESTING),
        ]
        fxt_assigner.assign.return_value = expected_assignments

        # Act
        getitune_trainer.assign_subsets(training_config=training_config, project_id=project_id)

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id)
        fxt_assigner.assign.assert_called_once()
        fxt_subset_service.update_subset_assignments.assert_called_once_with(project_id, expected_assignments)

    def test_assign_subsets_with_no_unassigned_items(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_subset_service: Mock,
        fxt_assigner: Mock,
    ):
        """Test assigning subsets when no unassigned items exist."""
        # Arrange
        project_id = uuid4()
        fxt_subset_service.get_unassigned_items_with_labels.return_value = []
        getitune_trainer = fxt_getitune_trainer()
        training_config = Mock(spec=TrainingConfiguration)

        # Act
        getitune_trainer.assign_subsets(training_config=training_config, project_id=project_id)

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id)
        fxt_assigner.assign.assert_not_called()
        fxt_subset_service.update_subset_assignments.assert_not_called()

    @pytest.mark.parametrize("has_all_subsets_assigned", [True, False], ids=["all-assigned", "not-all-assigned"])
    def test_assign_subsets_passes_has_all_subsets_assigned_to_assigner(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_subset_service: Mock,
        fxt_assigner: Mock,
        has_all_subsets_assigned: bool,
    ):
        """Test that the result of SubsetService.has_all_subsets_assigned is forwarded to SubsetAssigner.assign."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        project_id = uuid4()
        label = uuid4()

        unassigned_items = [
            DatasetItemWithLabels(item_id=uuid4(), labels={label}),
            DatasetItemWithLabels(item_id=uuid4(), labels={label}),
            DatasetItemWithLabels(item_id=uuid4(), labels={label}),
        ]
        fxt_subset_service.get_unassigned_items_with_labels.return_value = unassigned_items
        fxt_subset_service.has_all_subsets_assigned.return_value = has_all_subsets_assigned
        fxt_assigner.assign.return_value = [
            SubsetAssignment(item_id=item.item_id, subset=DatasetItemSubset.TRAINING) for item in unassigned_items
        ]

        training_config = TrainingConfiguration(
            task_level_parameters=TaskLevelParameters(),
            algo_level_parameters=MagicMock(spec=AlgoLevelParameters),
        )

        # Act
        getitune_trainer.assign_subsets(training_config=training_config, project_id=project_id)

        # Assert - has_all_subsets_assigned is queried from the service ...
        fxt_subset_service.has_all_subsets_assigned.assert_called_once_with(project_id)

        # … and the returned value is forwarded verbatim as the third positional argument
        call_args = fxt_assigner.assign.call_args
        actual_flag = (
            call_args.args[2] if len(call_args.args) >= 3 else call_args.kwargs.get("has_all_subsets_assigned")
        )
        assert actual_flag is has_all_subsets_assigned


class TestGetiTuneTrainerCreateTrainingDataset:
    """Tests for the GetiTuneTrainer.prepare_training_dataset method."""

    @pytest.mark.parametrize(
        "dataset_revision_id,uptodate_existing_dataset",
        [
            (None, False),
            (None, True),
            (uuid4(), False),
        ],
        ids=[
            "with new dataset revision",
            "with up-to-date existing dataset revision",
            "with existing dataset revision",
        ],
    )
    def test_prepare_training_dataset_success(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_dataset_service: Mock,
        fxt_dataset_revision_service: Mock,
        dataset_revision_id,
        uptodate_existing_dataset,
    ):
        """Test successful creation of training, validation, and testing datasets."""
        # Arrange
        project_id = uuid4()
        task = Task(task_type=TaskType.DETECTION, exclusive_labels=True)
        getitune_trainer = fxt_getitune_trainer()

        # Mock the Datumaro dataset
        mock_dm_dataset = Mock()
        fxt_dataset_service.get_dm_dataset.return_value = mock_dm_dataset
        fxt_dataset_revision_service.load_revision.return_value = mock_dm_dataset

        # Mock filtered subsets
        mock_training_subset = Mock()
        mock_validation_subset = Mock()
        mock_testing_subset = Mock()

        mock_dm_dataset.filter_by_subset.side_effect = [
            mock_training_subset,
            mock_validation_subset,
            mock_testing_subset,
        ]

        # Mock dataset revision saving
        new_dataset_revision_id = uuid4()  # this ID is only used when a new revision is created
        fxt_dataset_revision_service.save_revision.return_value = new_dataset_revision_id

        # Mock get_latest_uptodate_dataset_revision
        uptodate_revision = Mock()
        uptodate_revision.id = uuid4()
        if dataset_revision_id is None and uptodate_existing_dataset:
            fxt_dataset_revision_service.get_latest_uptodate_dataset_revision.return_value = uptodate_revision
        else:
            fxt_dataset_revision_service.get_latest_uptodate_dataset_revision.return_value = None

        # Create a getitune training configuration dict matching the expected structure
        getitune_training_config = {
            "data": {
                "input_size": (640, 640),
                "train_subset": {
                    "batch_size": 8,
                    "num_workers": 4,
                    "sampler": {"class_path": "torch.utils.data.RandomSampler"},
                    "augmentations_cpu": [
                        {"class_path": "torchvision.transforms.v2.RandomHorizontalFlip", "init_args": {"p": 0.5}}
                    ],
                },
                "val_subset": {
                    "batch_size": 4,
                    "num_workers": 2,
                    "sampler": {"class_path": "torch.utils.data.RandomSampler"},
                    "augmentations_cpu": [],
                },
                "test_subset": {
                    "batch_size": 2,
                    "num_workers": 1,
                    "sampler": {"class_path": "torch.utils.data.RandomSampler"},
                    "augmentations_cpu": [],
                },
            }
        }

        # Create a TrainingConfiguration with filtering disabled (default)
        training_config = TrainingConfiguration(
            task_level_parameters=TaskLevelParameters(),
            algo_level_parameters=MagicMock(spec=AlgoLevelParameters),
        )

        # Mock TransformLibFactory.generate to return mock transforms
        mock_train_transforms = [Mock()]
        mock_val_transforms = [Mock()]
        mock_test_transforms = [Mock()]

        with (
            patch("getitune.data.factory.TransformLibFactory.generate") as mock_generate,
            patch("getitune.data.entity.utils.detect_storage_dtype", return_value=("uint8", 3)),
        ):
            mock_generate.side_effect = [mock_train_transforms, mock_val_transforms, mock_test_transforms]

            # Mock the get_getitune_dataset_class_by_task_type function to return a proper mock class
            mock_dataset_class = Mock()
            mock_dataset_class.__name__ = "DetectionDataset"

            # Datasets need a working __len__ because prepare_training_dataset logs their sizes.
            mock_getitune_training_dataset = MagicMock()
            mock_getitune_training_dataset.__len__.return_value = 10
            mock_getitune_validation_dataset = MagicMock()
            mock_getitune_validation_dataset.__len__.return_value = 5
            mock_getitune_testing_dataset = MagicMock()
            mock_getitune_testing_dataset.__len__.return_value = 3

            mock_dataset_class.side_effect = [
                mock_getitune_training_dataset,
                mock_getitune_validation_dataset,
                mock_getitune_testing_dataset,
            ]

            with patch(
                "app.execution.training.getitune_trainer.get_getitune_dataset_class_by_task_type",
                return_value=mock_dataset_class,
            ):
                # Act
                dataset_info = getitune_trainer.prepare_training_dataset(
                    project_id=project_id,
                    task=task,
                    getitune_training_config=getitune_training_config,
                    training_config=training_config,
                    dataset_revision_id=dataset_revision_id,
                )

        # Assert
        # Verify that a dataset revision was created and saved if and only if no revision ID was provided
        if dataset_revision_id is None:
            if uptodate_existing_dataset:
                fxt_dataset_revision_service.get_latest_uptodate_dataset_revision.assert_called_once_with(
                    project_id=project_id
                )
                fxt_dataset_revision_service.load_revision.assert_called_once_with(
                    project_id=project_id, dataset_revision_id=uptodate_revision.id
                )
                fxt_dataset_service.get_dm_dataset.assert_not_called()
                fxt_dataset_revision_service.save_revision.assert_not_called()
                assert dataset_info.revision_id == uptodate_revision.id
            else:
                fxt_dataset_service.get_dm_dataset.assert_called_once_with(
                    project_id=project_id,
                    task=task,
                    annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
                    sample_mode=SampleMode.TRAINING,
                )
                fxt_dataset_revision_service.save_revision.assert_called_once_with(
                    project_id=project_id,
                    dataset=mock_dm_dataset,
                )
                fxt_dataset_revision_service.load_revision.assert_not_called()
                assert dataset_info.revision_id == new_dataset_revision_id
        else:
            fxt_dataset_service.get_dm_dataset.assert_not_called()
            fxt_dataset_revision_service.save_revision.assert_not_called()
            fxt_dataset_revision_service.load_revision.assert_called_once_with(
                project_id=project_id, dataset_revision_id=dataset_revision_id
            )
            assert dataset_info.revision_id == dataset_revision_id

        # Verify subsets were filtered for train, val, and test
        assert mock_dm_dataset.filter_by_subset.call_count == 3

        # Verify transforms were generated for each subset
        assert mock_generate.call_count == 3

        # Verify VisionDataset was instantiated three times with correct parameters
        assert mock_dataset_class.call_count == 3

        # Verify first call (training dataset)
        train_call = mock_dataset_class.call_args_list[0]
        assert train_call.kwargs["dm_subset"] == mock_training_subset
        assert train_call.kwargs["transforms"] == mock_train_transforms

        # Verify second call (validation dataset)
        val_call = mock_dataset_class.call_args_list[1]
        assert val_call.kwargs["dm_subset"] == mock_validation_subset
        assert val_call.kwargs["transforms"] == mock_val_transforms

        # Verify third call (testing dataset)
        test_call = mock_dataset_class.call_args_list[2]
        assert test_call.kwargs["dm_subset"] == mock_testing_subset
        assert test_call.kwargs["transforms"] == mock_test_transforms

        # Verify the returned DatasetInfo
        assert dataset_info.getitune_training_dataset == mock_getitune_training_dataset
        assert dataset_info.getitune_validation_dataset == mock_getitune_validation_dataset
        assert dataset_info.getitune_testing_dataset == mock_getitune_testing_dataset

        # Verify SubsetConfig objects were created correctly
        assert dataset_info.getitune_training_subset_config.batch_size == 8
        assert dataset_info.getitune_training_subset_config.num_workers == 4
        # pyrefly: ignore[missing-attribute]
        assert dataset_info.getitune_training_subset_config.transforms == mock_train_transforms

        assert dataset_info.getitune_validation_subset_config.batch_size == 4
        assert dataset_info.getitune_validation_subset_config.num_workers == 2
        # pyrefly: ignore[missing-attribute]
        assert dataset_info.getitune_validation_subset_config.transforms == mock_val_transforms

        assert dataset_info.getitune_testing_subset_config.batch_size == 2
        assert dataset_info.getitune_testing_subset_config.num_workers == 1
        # pyrefly: ignore[missing-attribute]
        assert dataset_info.getitune_testing_subset_config.transforms == mock_test_transforms


class TestGetiTuneTrainerPrepareModel:
    """Tests for the GetiTuneTrainer.prepare_model method."""

    def test_prepare_model(
        self,
        tmp_path: Path,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_model_service: Mock,
    ):
        """Test successful preparation of model metadata."""
        # Arrange
        project_id = uuid4()
        model_id = uuid4()
        model_architecture_id = "image-classification-efficientnet-b0"
        training_params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            model_id=model_id,
            project_id=project_id,
            model_architecture_id=model_architecture_id,
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
            parent_model_revision_id=None,
            job_id=uuid4(),
        )
        dataset_revision_id = uuid4()
        getitune_trainer = fxt_getitune_trainer()
        training_config = Mock(spec=TrainingConfiguration)

        # Act
        getitune_trainer.prepare_model(training_params, dataset_revision_id, training_config)

        # Assert
        fxt_model_service.create_revision.assert_called_once_with(
            ModelRevisionMetadata(
                model_id=model_id,
                model_name=training_params.model_name,
                project_id=project_id,
                architecture_id=model_architecture_id,
                parent_revision_id=None,
                training_status=TrainingStatus.NOT_STARTED,
                dataset_revision_id=dataset_revision_id,
                training_configuration=training_config,
            )
        )


class TestGetiTuneTrainerTrainModel:
    """Tests for the GetiTuneTrainer.train_model method."""

    @pytest.mark.parametrize(
        "geti_device,getitune_device",
        [
            (DeviceInfo(type=DeviceType.CPU, name="Intel Core", index=None, memory=None), "cpu"),
            (DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", index=0, memory=123), "xpu"),
            (DeviceInfo(type=DeviceType.CUDA, name="NVIDIA RTX 3090", index=1, memory=123), "gpu"),
        ],
        ids=["CPU", "XPU (Intel GPU)", "CUDA (NVIDIA GPU)"],
    )
    @pytest.mark.parametrize("add_precision", [True, False])
    def test_train_model(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        tmp_path: Path,
        geti_device: DeviceInfo,
        getitune_device: str,
        add_precision: bool,
    ):
        """Test successful model training."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        model_id = uuid4()

        # Mock DatasetInfo
        mock_dataset_info = Mock()
        mock_training_dataset = Mock()
        mock_validation_dataset = Mock()
        mock_testing_dataset = Mock()
        mock_training_subset_config = Mock()
        mock_validation_subset_config = Mock()
        mock_testing_subset_config = Mock()

        mock_dataset_info.getitune_training_dataset = mock_training_dataset
        mock_dataset_info.getitune_validation_dataset = mock_validation_dataset
        mock_dataset_info.getitune_testing_dataset = mock_testing_dataset
        mock_dataset_info.getitune_training_subset_config = mock_training_subset_config
        mock_dataset_info.getitune_validation_subset_config = mock_validation_subset_config
        mock_dataset_info.getitune_testing_subset_config = mock_testing_subset_config

        # Mock weights path
        weights_path = tmp_path / "weights.pth"
        weights_path.touch()

        # Create training configuration
        training_config: dict[str, Any] = {
            "model": {
                "class_path": "getitune.backend.lightning.models.detection.yolox.YOLOXModel",
                "init_args": {
                    "model_name": "yolox_tiny",
                },
            },
            "max_epochs": 10,
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": "./checkpoints",
                        "filename": "best_checkpoint",
                    },
                }
            ],
        }
        if add_precision:
            training_config["precision"] = "32"

        # Mock DataModule
        mock_datamodule = Mock()
        mock_datamodule.label_info.label_names = ["cat", "dog"]
        mock_datamodule.input_size = (640, 640)
        mock_datamodule.input_mean = [0.485, 0.456, 0.406]
        mock_datamodule.input_std = [0.229, 0.224, 0.225]
        mock_datamodule.input_intensity_config = IntensityConfig(
            storage_dtype="uint16",
            mode="range_scale",
            min_value=500,
            max_value=1200,
            scale_factor=0.5,
        )
        mock_datamodule.tile_config = None

        # Mock model and engine
        mock_getitune_model = Mock(spec=[])
        mock_getitune_engine = Mock()
        mock_getitune_engine.work_dir = str(tmp_path / f"getitune-workspace-{model_id}")
        Path(mock_getitune_engine.work_dir).mkdir(parents=True)

        # Create expected checkpoint file
        expected_checkpoint_path = Path(mock_getitune_engine.work_dir) / "best_checkpoint.pt"
        expected_checkpoint_path.touch()
        mock_getitune_engine.best_checkpoint = expected_checkpoint_path

        with (
            patch(
                "getitune.data.module.DataModule.from_vision_datasets",
                return_value=mock_datamodule,
            ) as mock_datamodule_factory,
            patch("app.execution.training.getitune_trainer.ArgumentParser") as mock_parser_cls,
            patch(
                "getitune.engine.create_engine",
                return_value=mock_getitune_engine,
            ) as mock_create_engine,
        ):
            mock_model_parser = Mock()
            mock_callbacks_parser = Mock()
            mock_parser_cls.side_effect = [mock_model_parser, mock_callbacks_parser]
            mock_model_parser.instantiate_classes.return_value.get.return_value = mock_getitune_model
            mock_callbacks_parser.instantiate_classes.return_value.get.return_value = []
            # Act
            trained_model_path, returned_engine = getitune_trainer.train_model(
                training_config=training_config,
                dataset_info=mock_dataset_info,
                weights_path=weights_path,
                model_id=model_id,
                device=geti_device,
                has_model_revision=False,
            )

        # Assert
        # Verify DataModule was created correctly
        mock_datamodule_factory.assert_called_once_with(
            train_dataset=mock_training_dataset,
            val_dataset=mock_validation_dataset,
            test_dataset=mock_testing_dataset,
            train_subset=mock_training_subset_config,
            val_subset=mock_validation_subset_config,
            test_subset=mock_testing_subset_config,
        )

        # Verify engine was created via create_engine
        mock_create_engine.assert_called_once()
        engine_call_kwargs = mock_create_engine.call_args.kwargs
        assert engine_call_kwargs["model"] == mock_getitune_model
        assert engine_call_kwargs["data"] == mock_datamodule
        assert engine_call_kwargs["work_dir"] == getitune_trainer._data_dir / f"getitune-workspace-{model_id}"
        assert "checkpoint" not in engine_call_kwargs

        # Verify training was started
        mock_getitune_engine.train.assert_called_once()
        train_call_kwargs = mock_getitune_engine.train.call_args.kwargs
        assert train_call_kwargs["max_epochs"] == 10
        if add_precision:
            assert train_call_kwargs["precision"] == "32"
        else:
            assert "precision" not in train_call_kwargs
        if geti_device.type == DeviceType.CPU or geti_device.index is None:
            assert "devices" not in train_call_kwargs
        else:
            assert train_call_kwargs["devices"] == [geti_device.index]
        assert "callbacks" in train_call_kwargs

        # Verify return values
        assert trained_model_path == expected_checkpoint_path
        assert returned_engine == mock_getitune_engine

    def test_train_ultralytics_model_uses_unified_path(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        tmp_path: Path,
    ):
        """Test Ultralytics model training goes through the unified train_model path."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        model_id = uuid4()
        weights_path = tmp_path / "model.pt"
        weights_path.touch()
        expected_checkpoint_path = tmp_path / "best.pt"
        expected_checkpoint_path.touch()
        training_config: dict[str, Any] = {
            "backend": "ultralytics",
            "task": "DETECTION",
            "max_epochs": 3,
            "model": {
                "class_path": "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel",
                "init_args": {"model_name": "yolo26n.yaml"},
            },
            "training": {"epochs": 3, "batch": 2},
        }
        mock_dataset_info = Mock()
        mock_dataset_info.getitune_training_dataset = Mock()
        mock_dataset_info.getitune_validation_dataset = Mock()
        mock_dataset_info.getitune_testing_dataset = Mock()
        mock_dataset_info.getitune_training_subset_config = Mock()
        mock_dataset_info.getitune_validation_subset_config = Mock()
        mock_dataset_info.getitune_testing_subset_config = Mock()
        mock_datamodule = Mock()
        mock_datamodule.label_info = Mock()
        mock_datamodule.label_info.label_names = ["cls_a", "cls_b"]
        mock_datamodule.input_size = (640, 640)
        mock_datamodule.input_mean = (0.0, 0.0, 0.0)
        mock_datamodule.input_std = (1.0, 1.0, 1.0)
        mock_datamodule.input_intensity_config = None
        mock_datamodule.tile_config = None
        mock_model = Mock()
        mock_model.load_checkpoint = Mock()
        mock_engine = Mock()
        mock_engine.best_checkpoint = expected_checkpoint_path
        mock_engine.work_dir = tmp_path

        with (
            patch("getitune.data.module.DataModule.from_vision_datasets", return_value=mock_datamodule),
            patch("app.execution.training.getitune_trainer.ArgumentParser") as mock_parser_cls,
            patch("getitune.engine.create_engine", return_value=mock_engine) as mock_create_engine,
        ):
            mock_model_parser = Mock()
            mock_callbacks_parser = Mock()
            mock_parser_cls.side_effect = [mock_model_parser, mock_callbacks_parser]
            mock_model_parser.instantiate_classes.return_value.get.return_value = mock_model
            mock_callbacks_parser.instantiate_classes.return_value.get.return_value = []
            # Act
            trained_model_path, returned_engine = getitune_trainer.train_model(
                training_config=training_config,
                dataset_info=mock_dataset_info,
                weights_path=weights_path,
                model_id=model_id,
                device=DeviceInfo(type=DeviceType.CPU, name="Intel Core", index=None, memory=None),
                has_model_revision=True,
            )

        # Assert
        mock_create_engine.assert_called_once()
        engine_call_kwargs = mock_create_engine.call_args.kwargs
        assert engine_call_kwargs["checkpoint"] == weights_path
        mock_engine.train.assert_called_once()
        assert trained_model_path == expected_checkpoint_path
        assert returned_engine == mock_engine


class TestGetiTuneTrainerExecuteCancellation:
    """Tests for the GetiTuneTrainer.execute method handling CancelledExc."""

    def test_execute_cancellation_during_training_deletes_model_revision(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_model_service: Mock,
    ):
        """
        When CancelledExc is raised during train_model, the model revision should be deleted and the
        exception re-raised.
        """
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        project_id = uuid4()
        model_id = uuid4()
        dataset_revision_id = uuid4()

        params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            project_id=project_id,
            model_id=model_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=None,
            job_id=uuid4(),
        )

        mock_training_config = Mock(spec=TrainingConfiguration)
        mock_getitune_training_config = {"key": "value"}
        mock_dataset_info = Mock(spec=DatasetInfo)
        mock_dataset_info.revision_id = dataset_revision_id
        mock_weights_path = Path("/fake/weights.pth")

        # Stub all steps that run before the try block
        with (
            patch.object(getitune_trainer, "prepare_weights", return_value=mock_weights_path),
            patch.object(
                getitune_trainer,
                "prepare_training_configuration",
                return_value=(mock_training_config, mock_getitune_training_config),
            ),
            patch.object(getitune_trainer, "assign_subsets"),
            patch.object(getitune_trainer, "prepare_training_dataset", return_value=mock_dataset_info),
            patch.object(getitune_trainer, "prepare_model"),
            patch.object(getitune_trainer, "train_model", side_effect=CancelledExc("Job cancelled")),
            pytest.raises(CancelledExc),
        ):
            # Act & Assert
            getitune_trainer.execute(params)

        # Assert - model revision was deleted
        fxt_model_service.delete_model.assert_called_once_with(project_id=project_id, model_id=model_id)

    def test_execute_cancellation_during_export_deletes_model_revision(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_model_service: Mock,
    ):
        """When CancelledExc is raised during export_model, the model revision should be deleted."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        project_id = uuid4()
        model_id = uuid4()
        dataset_revision_id = uuid4()

        params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            project_id=project_id,
            model_id=model_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=None,
            job_id=uuid4(),
        )

        mock_training_config = Mock(spec=TrainingConfiguration)
        mock_getitune_training_config = {"key": "value"}
        mock_dataset_info = Mock(spec=DatasetInfo)
        mock_dataset_info.revision_id = dataset_revision_id
        mock_weights_path = Path("/fake/weights.pth")
        mock_getitune_engine = Mock()
        mock_trained_model_path = Path("/fake/best_checkpoint.pt")

        with (
            patch.object(getitune_trainer, "prepare_weights", return_value=mock_weights_path),
            patch.object(
                getitune_trainer,
                "prepare_training_configuration",
                return_value=(mock_training_config, mock_getitune_training_config),
            ),
            patch.object(getitune_trainer, "assign_subsets"),
            patch.object(getitune_trainer, "prepare_training_dataset", return_value=mock_dataset_info),
            patch.object(getitune_trainer, "prepare_model"),
            patch.object(getitune_trainer, "train_model", return_value=(mock_trained_model_path, mock_getitune_engine)),
            patch.object(getitune_trainer, "export_model", side_effect=CancelledExc("Job cancelled")),
            pytest.raises(CancelledExc),
        ):
            getitune_trainer.execute(params)

        fxt_model_service.delete_model.assert_called_once_with(project_id=project_id, model_id=model_id)

    def test_execute_failure_during_training_marks_model_as_failed(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_model_service: Mock,
    ):
        """When a non-cancellation exception is raised during train_model, the model should be marked FAILED."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        project_id = uuid4()
        model_id = uuid4()
        dataset_revision_id = uuid4()

        params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            project_id=project_id,
            model_id=model_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=None,
            job_id=uuid4(),
        )

        mock_training_config = Mock(spec=TrainingConfiguration)
        mock_getitune_training_config = {"key": "value"}
        mock_dataset_info = Mock(spec=DatasetInfo)
        mock_dataset_info.revision_id = dataset_revision_id
        mock_weights_path = Path("/fake/weights.pth")

        with (
            patch.object(getitune_trainer, "prepare_weights", return_value=mock_weights_path),
            patch.object(
                getitune_trainer,
                "prepare_training_configuration",
                return_value=(mock_training_config, mock_getitune_training_config),
            ),
            patch.object(getitune_trainer, "assign_subsets"),
            patch.object(getitune_trainer, "prepare_training_dataset", return_value=mock_dataset_info),
            patch.object(getitune_trainer, "prepare_model"),
            patch.object(getitune_trainer, "train_model", side_effect=RuntimeError("getitune crashed")),
            pytest.raises(RuntimeError, match="getitune crashed"),
        ):
            getitune_trainer.execute(params)

        # Assert - model should be marked FAILED, NOT deleted
        fxt_model_service.delete_model.assert_not_called()
        status_calls = fxt_model_service.update_revision_status.call_args_list
        last_status_call = status_calls[-1]
        assert last_status_call.kwargs.get("training_status") == TrainingStatus.FAILED

        # Assert - the IN_PROGRESS update recorded which hardware was used to run the training
        in_progress_call = next(
            call for call in status_calls if call.kwargs.get("training_status") == TrainingStatus.IN_PROGRESS
        )
        assert in_progress_call.kwargs.get("training_device") == params.device


class TestGetiTuneTrainerEvaluateModel:
    """Tests for the GetiTuneTrainer.evaluate_model method."""

    @pytest.mark.parametrize(
        "task_type,exclusive_labels,metric_callable",
        [
            (TaskType.CLASSIFICATION, True, MultiClassClsMetricCallable),
            (TaskType.CLASSIFICATION, False, MultiLabelClsMetricCallable),
            (TaskType.DETECTION, False, MeanAPCallable),
            (TaskType.INSTANCE_SEGMENTATION, False, MaskRLEMeanAPCallable),
        ],
    )
    def test_evaluate_model(
        self,
        task_type: TaskType,
        exclusive_labels: bool,
        metric_callable: MetricCallable,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        tmp_path: Path,
        fxt_model_service: Mock,
    ):
        """Test successful evaluation of all three model variants (PyTorch, OV, ONNX)."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        project_id = uuid4()
        model_id = uuid4()
        pytorch_variant_id = uuid4()
        openvino_variant_id = uuid4()
        onnx_variant_id = uuid4()
        dataset_revision_id = uuid4()

        pytorch_metrics = {
            "test/accuracy": torch.tensor(0.85),
            "test/precision": torch.tensor(0.82),
            "test/recall": torch.tensor(0.88),
        }
        ov_metrics = {
            "test/accuracy": torch.tensor(0.84),
            "test/precision": torch.tensor(0.81),
            "test/recall": torch.tensor(0.87),
        }
        onnx_metrics = {
            "test/accuracy": torch.tensor(0.83),
            "test/precision": torch.tensor(0.80),
            "test/recall": torch.tensor(0.86),
        }

        mock_getitune_engine = Mock()
        mock_getitune_engine.test.return_value = pytorch_metrics
        mock_getitune_engine.work_dir = tmp_path / "getitune-workspace"
        mock_getitune_engine.datamodule = Mock()

        model_checkpoint_path = tmp_path / "best_checkpoint.pt"
        model_checkpoint_path.touch()
        ov_export_path = tmp_path / "exported_model"
        onnx_export_path = tmp_path / "exported_model"
        ov_xml_path = ov_export_path.with_suffix(".xml")
        onnx_path = onnx_export_path.with_suffix(".onnx")

        model_variants = [
            ModelVariantDescriptor(
                id=pytorch_variant_id,
                path=model_checkpoint_path,
                format=ModelFormat.PYTORCH,
            ),
            ModelVariantDescriptor(
                id=openvino_variant_id,
                path=ov_xml_path,
                format=ModelFormat.OPENVINO,
            ),
            ModelVariantDescriptor(
                id=onnx_variant_id,
                path=onnx_path,
                format=ModelFormat.ONNX,
            ),
        ]

        training_params = TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            model_id=model_id,
            project_id=project_id,
            model_architecture_id="object-detection-yolox-s",
            model_architecture_name="Test Model",
            task=Task(task_type=task_type, exclusive_labels=exclusive_labels),
            job_id=uuid4(),
        )

        mock_ov_engine = Mock()
        mock_ov_engine.test.return_value = ov_metrics
        mock_onnx_engine = Mock()
        mock_onnx_engine.test.return_value = onnx_metrics

        # Act
        with patch(
            "getitune.backend.openvino.engine.OVEngine",
            side_effect=[mock_ov_engine, mock_onnx_engine],
        ) as mock_ov_engine_cls:
            getitune_trainer.evaluate_model(
                getitune_engine=mock_getitune_engine,
                task=training_params.task,
                model_revision_id=model_id,
                model_variants=model_variants,
                dataset_revision_id=dataset_revision_id,
            )

        # Assert: PyTorch evaluation via the LightningEngine
        mock_getitune_engine.test.assert_called_once_with(metric=metric_callable)

        # Assert: OVEngine instantiated twice (OV + ONNX) and tested with the right checkpoints
        assert mock_ov_engine_cls.call_count == 2
        mock_ov_engine_cls.assert_has_calls(
            calls=[
                call(
                    model=ov_xml_path,
                    data=mock_getitune_engine.datamodule,
                    work_dir=mock_getitune_engine.work_dir / "ov_eval",
                ),
                call(
                    model=onnx_path,
                    data=mock_getitune_engine.datamodule,
                    work_dir=mock_getitune_engine.work_dir / "onnx_eval",
                ),
            ]
        )
        mock_ov_engine.test.assert_called_once_with(metric=metric_callable)
        mock_onnx_engine.test.assert_called_once_with(metric=metric_callable)

        # Assert: each variant's metrics are persisted with the matching variant id
        save_calls = fxt_model_service.save_evaluation_result.call_args_list
        assert len(save_calls) == 3

        results_by_variant = {c.args[0].model_variant_id: c.args[0] for c in save_calls}
        assert set(results_by_variant) == {pytorch_variant_id, openvino_variant_id, onnx_variant_id}

        for result in results_by_variant.values():
            assert result.model_revision_id == model_id
            assert result.dataset_revision_id == dataset_revision_id
            assert result.subset == DatasetItemSubset.TESTING

        assert results_by_variant[pytorch_variant_id].metrics == pytest.approx(
            {"accuracy": 0.85, "precision": 0.82, "recall": 0.88}, rel=1e-6
        )
        assert results_by_variant[openvino_variant_id].metrics == pytest.approx(
            {"accuracy": 0.84, "precision": 0.81, "recall": 0.87}, rel=1e-6
        )
        assert results_by_variant[onnx_variant_id].metrics == pytest.approx(
            {"accuracy": 0.83, "precision": 0.80, "recall": 0.86}, rel=1e-6
        )

    def test_evaluate_ultralytics_model_without_lightning_metric(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        tmp_path: Path,
        fxt_model_service: Mock,
    ):
        """Test Ultralytics evaluation passes metric uniformly (engine ignores it)."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        model_id = uuid4()
        pytorch_variant_id = uuid4()
        dataset_revision_id = uuid4()
        mock_getitune_engine = Mock()
        mock_getitune_engine.test = Mock(return_value={"metrics/mAP50-95(B)": torch.tensor(0.42)})
        mock_getitune_engine.work_dir = tmp_path / "getitune-workspace"
        mock_getitune_engine.datamodule = Mock()
        model_checkpoint_path = tmp_path / "best.pt"
        model_checkpoint_path.touch()

        model_variants = [
            ModelVariantDescriptor(
                id=pytorch_variant_id,
                path=model_checkpoint_path,
                format=ModelFormat.PYTORCH,
            ),
        ]

        # Act
        getitune_trainer.evaluate_model(
            getitune_engine=mock_getitune_engine,
            task=Task(task_type=TaskType.DETECTION),
            model_revision_id=model_id,
            model_variants=model_variants,
            dataset_revision_id=dataset_revision_id,
        )

        # Assert — metric is always passed uniformly; Ultralytics engines ignore it.
        mock_getitune_engine.test.assert_called_once_with(metric=MeanAPCallable)
        fxt_model_service.save_evaluation_result.assert_called_once()

    def test_evaluate_model_reuses_pytorch_results_when_openvino_fails(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        tmp_path: Path,
        fxt_model_service: Mock,
    ):
        """OpenVINO/ONNX evaluation failures must not fail the job; PyTorch results are reused instead."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        model_id = uuid4()
        pytorch_variant_id = uuid4()
        openvino_variant_id = uuid4()
        onnx_variant_id = uuid4()
        dataset_revision_id = uuid4()

        pytorch_metrics = {"test/accuracy": torch.tensor(0.85)}

        mock_getitune_engine = Mock()
        mock_getitune_engine.test.return_value = pytorch_metrics
        mock_getitune_engine.work_dir = tmp_path / "getitune-workspace"
        mock_getitune_engine.datamodule = Mock()

        model_checkpoint_path = tmp_path / "best_checkpoint.pt"
        model_checkpoint_path.touch()
        ov_xml_path = tmp_path / "exported_model.xml"
        onnx_path = tmp_path / "exported_model.onnx"

        model_variants = [
            ModelVariantDescriptor(id=pytorch_variant_id, path=model_checkpoint_path, format=ModelFormat.PYTORCH),
            ModelVariantDescriptor(id=openvino_variant_id, path=ov_xml_path, format=ModelFormat.OPENVINO),
            ModelVariantDescriptor(id=onnx_variant_id, path=onnx_path, format=ModelFormat.ONNX),
        ]

        mock_ov_engine = Mock()
        mock_ov_engine.test.side_effect = RuntimeError("OpenVINO evaluation blew up")
        mock_onnx_engine = Mock()
        mock_onnx_engine.test.return_value = {"test/accuracy": torch.tensor(0.83)}

        # Act
        with (
            patch(
                "getitune.backend.openvino.engine.OVEngine",
                side_effect=[mock_ov_engine, mock_onnx_engine],
            ),
            patch("app.execution.training.getitune_trainer.logger") as mock_logger,
        ):
            getitune_trainer.evaluate_model(
                getitune_engine=mock_getitune_engine,
                task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
                model_revision_id=model_id,
                model_variants=model_variants,
                dataset_revision_id=dataset_revision_id,
            )

        # Assert: a warning was logged for the failed OpenVINO evaluation
        mock_logger.warning.assert_called_once()
        assert "openvino" in mock_logger.warning.call_args.args[1].lower()

        # Assert: all three variants still got a saved evaluation result
        save_calls = fxt_model_service.save_evaluation_result.call_args_list
        assert len(save_calls) == 3
        results_by_variant = {c.args[0].model_variant_id: c.args[0] for c in save_calls}

        # The OpenVINO variant reused the PyTorch metrics instead of failing
        assert results_by_variant[openvino_variant_id].metrics == results_by_variant[pytorch_variant_id].metrics
        # The ONNX variant still evaluated normally and kept its own metrics
        assert results_by_variant[onnx_variant_id].metrics == pytest.approx({"accuracy": 0.83}, rel=1e-6)

    def test_evaluate_model_raises_when_pytorch_evaluation_fails(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        tmp_path: Path,
        fxt_model_service: Mock,
    ):
        """A failure evaluating the PyTorch variant itself must still fail the job (no fallback available)."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        model_id = uuid4()
        pytorch_variant_id = uuid4()
        dataset_revision_id = uuid4()

        mock_getitune_engine = Mock()
        mock_getitune_engine.test.side_effect = RuntimeError("PyTorch evaluation blew up")
        mock_getitune_engine.work_dir = tmp_path / "getitune-workspace"
        mock_getitune_engine.datamodule = Mock()

        model_checkpoint_path = tmp_path / "best_checkpoint.pt"
        model_checkpoint_path.touch()

        model_variants = [
            ModelVariantDescriptor(id=pytorch_variant_id, path=model_checkpoint_path, format=ModelFormat.PYTORCH),
        ]

        # Act / Assert
        with pytest.raises(RuntimeError, match="PyTorch evaluation blew up"):
            getitune_trainer.evaluate_model(
                getitune_engine=mock_getitune_engine,
                task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
                model_revision_id=model_id,
                model_variants=model_variants,
                dataset_revision_id=dataset_revision_id,
            )
        fxt_model_service.save_evaluation_result.assert_not_called()


class TestGetiTuneTrainerExportModel:
    """Tests for the GetiTuneTrainer.export_model method."""

    def test_export_model(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        tmp_path: Path,
    ):
        """Test successful model export to OpenVINO and ONNX format."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        mock_getitune_engine = Mock()
        model_checkpoint_path = tmp_path / "best_checkpoint.pt"
        model_checkpoint_path.touch()
        expected_ov_export_path = tmp_path / "exported_openvino_model"
        expected_onnx_export_path = tmp_path / "exported_onnx_model"
        mock_getitune_engine.export.side_effect = [expected_ov_export_path, expected_onnx_export_path]

        # Act
        exported_paths = getitune_trainer.export_model(
            getitune_engine=mock_getitune_engine,
            model_checkpoint_path=model_checkpoint_path,
        )

        # Assert
        assert mock_getitune_engine.export.call_count == 2
        mock_getitune_engine.export.assert_has_calls(
            calls=[
                call(
                    checkpoint=model_checkpoint_path,
                    export_format=ExportFormat.OPENVINO,
                    export_precision=Precision.FP16,
                ),
                call(
                    checkpoint=model_checkpoint_path,
                    export_format=ExportFormat.ONNX,
                    export_precision=Precision.FP16,
                ),
            ]
        )
        assert exported_paths.openvino_model_path == expected_ov_export_path
        assert exported_paths.onnx_model_path == expected_onnx_export_path


class TestGetiTuneTrainerStoreModelArtifacts:
    """Tests for the GetiTuneTrainer.store_model_artifacts method."""

    def test_store_model_artifacts(
        self,
        fxt_getitune_trainer: Callable[[], GetiTuneTrainer],
        fxt_model_service: Mock,
        tmp_path: Path,
    ):
        """Test successful storing of model artifacts and cleanup."""
        # Arrange
        getitune_trainer = fxt_getitune_trainer()
        project_id = uuid4()
        model_id = uuid4()

        pytorch_variant_id = uuid4()
        openvino_variant_id = uuid4()
        onnx_variant_id = uuid4()

        # Pre-built mapping as returned by create_model_variants
        created_variants = {
            ModelFormat.PYTORCH: pytorch_variant_id,
            ModelFormat.OPENVINO: openvino_variant_id,
            ModelFormat.ONNX: onnx_variant_id,
        }

        # Create model directory structure
        model_dir = tmp_path / "projects" / str(project_id) / "models" / str(model_id)
        model_dir.mkdir(parents=True)

        # Create getitune work directory with artifacts
        getitune_work_dir = tmp_path / f"getitune-workspace-{model_id}"
        getitune_work_dir.mkdir(parents=True)

        # Create model checkpoint
        trained_model_path = getitune_work_dir / "best_checkpoint.pt"
        trained_model_path.write_text("checkpoint content")

        # Create exported model files
        exported_model_path = getitune_work_dir / "exported_model.pth"
        model_xml_path = exported_model_path.with_suffix(".xml")
        model_bin_path = exported_model_path.with_suffix(".bin")
        model_onnx_path = exported_model_path.with_suffix(".onnx")
        model_xml_path.write_text("xml content")
        model_bin_path.write_text("bin content")
        model_onnx_path.write_text("onnx content")

        exported_model_paths = ExportedModels(
            openvino_model_path=exported_model_path,
            onnx_model_path=exported_model_path,
        )

        # Create metrics directory
        metrics_dir = getitune_work_dir / "csv"
        metrics_dir.mkdir()
        (metrics_dir / "metrics.csv").write_text("epoch,loss\n1,0.5\n")

        # Act
        getitune_trainer.store_model_artifacts(
            model_dir=model_dir,
            getitune_work_dir=getitune_work_dir,
            trained_model_path=trained_model_path,
            exported_model_paths=exported_model_paths,
            created_variants=created_variants,
        )

        # Assert

        # Check variant directories and files
        variants_dir = model_dir / "variants"
        assert variants_dir.exists()

        pytorch_dir = variants_dir / str(pytorch_variant_id)
        assert pytorch_dir.exists()
        assert (pytorch_dir / "model.pt").exists()
        assert (pytorch_dir / "model.pt").read_text() == "checkpoint content"

        # Check OpenVINO variant
        openvino_dir = variants_dir / str(openvino_variant_id)
        assert openvino_dir.exists()
        assert (openvino_dir / "model.xml").exists()
        assert (openvino_dir / "model.xml").read_text() == "xml content"
        assert (openvino_dir / "model.bin").exists()
        assert (openvino_dir / "model.bin").read_text() == "bin content"

        # Check ONNX variant
        onnx_dir = variants_dir / str(onnx_variant_id)
        assert onnx_dir.exists()
        assert (onnx_dir / "model.onnx").exists()
        assert (onnx_dir / "model.onnx").read_text() == "onnx content"

        # Check metrics were moved
        assert (model_dir / "metrics").exists()
        assert (model_dir / "metrics" / "metrics.csv").exists()
        assert (model_dir / "metrics" / "metrics.csv").read_text() == "epoch,loss\n1,0.5\n"

        # The getitune work directory is no longer cleaned up here; it is removed
        # by ``TrainingJob.on_complete`` after the job finishes.
        assert getitune_work_dir.exists()


# ---------------------------------------------------------------------------
# Helpers shared across TestGetiTuneTrainerFilterDataset
# ---------------------------------------------------------------------------

_IMG_INFO = ImageInfo(width=100, height=100)
_DUMMY_IMAGE = LazyImage("/dummy/path/img.jpg")
# Four categories cover label indices 0-3, which is the maximum used across all test helpers.
_LABEL_CATEGORIES = LabelCategories(labels=("cat", "dog", "bird", "fish"))


def _make_detection_dataset(*bbox_counts: int, subset: Subset = Subset.TRAINING) -> Dataset[DetectionTrainingSample]:
    """Build a DetectionTrainingSample dataset where each sample has the given number of bboxes."""
    dataset: Dataset[DetectionTrainingSample] = Dataset(
        DetectionTrainingSample, categories={"label": _LABEL_CATEGORIES}
    )
    for count in bbox_counts:
        bboxes = np.array([[i * 10, i * 10, i * 10 + 5, i * 10 + 5] for i in range(count)])
        labels = np.array([0] * count)
        dataset.append(
            DetectionTrainingSample(
                id=str(uuid4()),
                image=_DUMMY_IMAGE,
                image_info=_IMG_INFO,
                subset=subset,
                bboxes=bboxes,
                label=labels,
                confidence=None,
            )
        )
    return dataset


def _make_instance_seg_dataset(
    *polygon_counts: int, subset: Subset = Subset.TRAINING
) -> Dataset[InstanceSegmentationTrainingSample]:
    """Build an InstanceSegmentationTrainingSample dataset where each sample has the given number of polygons."""
    dataset: Dataset[InstanceSegmentationTrainingSample] = Dataset(
        InstanceSegmentationTrainingSample, categories={"label": _LABEL_CATEGORIES}
    )
    for count in polygon_counts:
        # Each polygon is a simple square represented as 4 (x, y) vertices
        polygons = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] for _ in range(count)], dtype=np.float32)
        labels = np.array([0] * count)
        dataset.append(
            InstanceSegmentationTrainingSample(
                id=str(uuid4()),
                image=_DUMMY_IMAGE,
                image_info=_IMG_INFO,
                subset=subset,
                polygons=polygons,
                label=labels,
                confidence=None,
            )
        )
    return dataset


def _make_multiclass_dataset(
    *labels: int | None, subset: Subset = Subset.TRAINING
) -> Dataset[MulticlassClassificationTrainingSample]:
    """Build a MulticlassClassificationTrainingSample dataset with the given per-sample labels."""
    dataset: Dataset[MulticlassClassificationTrainingSample] = Dataset(
        MulticlassClassificationTrainingSample, categories={"label": _LABEL_CATEGORIES}
    )
    for lbl in labels:
        dataset.append(
            MulticlassClassificationTrainingSample(
                id=str(uuid4()),
                image=_DUMMY_IMAGE,
                image_info=_IMG_INFO,
                subset=subset,
                label=lbl,
                confidence=None,
            )
        )
    return dataset


def _make_multilabel_dataset(
    *label_lists: list[int], subset: Subset = Subset.TRAINING
) -> Dataset[MultilabelClassificationTrainingSample]:
    """Build a MultilabelClassificationTrainingSample dataset where each sample has the given label list."""
    dataset: Dataset[MultilabelClassificationTrainingSample] = Dataset(
        MultilabelClassificationTrainingSample, categories={"label": _LABEL_CATEGORIES}
    )
    for labels in label_lists:
        dataset.append(
            MultilabelClassificationTrainingSample(
                id=str(uuid4()),
                image=_DUMMY_IMAGE,
                image_info=_IMG_INFO,
                subset=subset,
                label=np.array(labels),
                confidence=None,
            )
        )
    return dataset


def _make_training_config(
    min_enabled: bool = False,
    min_value: int = 1,
    max_enabled: bool = False,
    max_value: int = 10,
) -> TrainingConfiguration:
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters.model_validate(
            {
                "dataset_preparation": {
                    "filtering": {
                        "min_annotation_objects": {"enable": min_enabled, "value": min_value},
                        "max_annotation_objects": {"enable": max_enabled, "value": max_value},
                    }
                }
            }
        ),
        algo_level_parameters=MagicMock(spec=AlgoLevelParameters),
    )


class TestGetiTuneTrainerFilterDataset:
    """Unit tests for GetiTuneTrainer._filter_dataset."""

    # ------------------------------------------------------------------
    # Filtering disabled
    # ------------------------------------------------------------------

    def test_returns_original_dataset_when_filtering_disabled(
        self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]
    ) -> None:
        """When both min and max are disabled the exact same dataset object is returned."""
        task = Task(task_type=TaskType.DETECTION)
        dataset = _make_detection_dataset(1, 3, 5)
        training_config = _make_training_config()  # both disabled

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert result is dataset

    # ------------------------------------------------------------------
    # Detection - bboxes field
    # ------------------------------------------------------------------

    def test_detection_min_filter_removes_empty_samples(
        self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]
    ) -> None:
        """Samples with 0 bboxes are removed when min=1 is enabled."""
        task = Task(task_type=TaskType.DETECTION)
        dataset = _make_detection_dataset(0, 1, 2, 3)
        training_config = _make_training_config(min_enabled=True, min_value=1)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 3

    def test_detection_max_filter_removes_crowded_samples(
        self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]
    ) -> None:
        """Samples exceeding the max bbox count are removed."""
        task = Task(task_type=TaskType.DETECTION)
        dataset = _make_detection_dataset(1, 2, 3, 4, 5)
        training_config = _make_training_config(max_enabled=True, max_value=3)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 3  # 1, 2, 3 bboxes pass; 4, 5 are removed

    def test_detection_min_and_max_filter_keeps_only_range(
        self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]
    ) -> None:
        """Both min and max enabled: only samples within [min, max] survive."""
        task = Task(task_type=TaskType.DETECTION)
        dataset = _make_detection_dataset(0, 1, 2, 3, 4, 5)
        training_config = _make_training_config(min_enabled=True, min_value=2, max_enabled=True, max_value=4)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 3  # 2, 3, 4 pass

    def test_detection_filter_preserves_schema(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        """The returned dataset has the same dtype and schema as the input."""
        task = Task(task_type=TaskType.DETECTION)
        dataset = _make_detection_dataset(1, 3, 5)
        training_config = _make_training_config(min_enabled=True, min_value=2)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert result.dtype == dataset.dtype
        assert result.schema == dataset.schema

    def test_detection_filter_all_removed_returns_empty_dataset(
        self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]
    ) -> None:
        """When every sample is filtered out the result is an empty (but valid) dataset."""
        task = Task(task_type=TaskType.DETECTION)
        dataset = _make_detection_dataset(1, 2, 3)
        training_config = _make_training_config(min_enabled=True, min_value=10)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 0

    # ------------------------------------------------------------------
    # Instance segmentation - polygons field
    # ------------------------------------------------------------------

    def test_instance_seg_min_filter(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        """Samples below the polygon minimum are removed."""
        task = Task(task_type=TaskType.INSTANCE_SEGMENTATION)
        dataset = _make_instance_seg_dataset(0, 1, 2, 3)
        training_config = _make_training_config(min_enabled=True, min_value=2)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 2  # 2, 3 polygons pass

    def test_instance_seg_max_filter(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        """Samples above the polygon maximum are removed."""
        task = Task(task_type=TaskType.INSTANCE_SEGMENTATION)
        dataset = _make_instance_seg_dataset(1, 2, 3, 4)
        training_config = _make_training_config(max_enabled=True, max_value=2)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 2  # 1, 2 polygons pass

    # ------------------------------------------------------------------
    # Multiclass classification - scalar label field
    # ------------------------------------------------------------------

    def test_multiclass_min_filter_removes_unlabelled_samples(
        self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]
    ) -> None:
        """Samples without a label (None) are removed when min is enabled."""
        task = Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True)
        dataset = _make_multiclass_dataset(None, 0, 1, None, 2)
        training_config = _make_training_config(min_enabled=True, min_value=1)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 3  # the three samples with an actual label survive

    def test_multiclass_max_filter_is_noop(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        """Max filtering is a no-op for multiclass classification (label is always scalar)."""
        task = Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True)
        dataset = _make_multiclass_dataset(0, 1, 2)
        training_config = _make_training_config(max_enabled=True, max_value=1)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        # No samples removed - max is meaningless for a scalar label
        assert len(result.df) == 3

    # ------------------------------------------------------------------
    # Multilabel classification - label list field
    # ------------------------------------------------------------------

    def test_multilabel_min_filter(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        """Samples with fewer labels than the minimum are removed."""
        task = Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=False)
        dataset = _make_multilabel_dataset([], [0], [0, 1], [0, 1, 2])
        training_config = _make_training_config(min_enabled=True, min_value=2)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 2  # [0, 1] and [0, 1, 2] pass

    def test_multilabel_max_filter(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        """Samples with more labels than the maximum are removed."""
        task = Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=False)
        dataset = _make_multilabel_dataset([0], [0, 1], [0, 1, 2])
        training_config = _make_training_config(max_enabled=True, max_value=2)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 2  # [0] and [0, 1] pass

    def test_multilabel_min_and_max_filter(self, fxt_getitune_trainer: Callable[[], GetiTuneTrainer]) -> None:
        """Both min and max applied simultaneously on multilabel classification."""
        task = Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=False)
        dataset = _make_multilabel_dataset([], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3])
        training_config = _make_training_config(min_enabled=True, min_value=1, max_enabled=True, max_value=2)

        result = fxt_getitune_trainer().filter_dataset(dm_dataset=dataset, task=task, training_config=training_config)

        assert len(result.df) == 2  # [0] and [0, 1] pass
