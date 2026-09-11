# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.base import DataFormat
from datumaro.experimental.fields import Subset

from app.datumaro_converter import SampleMode
from app.execution import ExportDataset
from app.models import (
    DatasetFormat,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    ExportDatasetJobParams,
    Task,
    TaskType,
)


@pytest.fixture
def fxt_project_service() -> Mock:
    return Mock()


@pytest.fixture
def fxt_export(
    fxt_staged_datasets_dir: Path,
    fxt_dataset_service: Mock,
    fxt_dataset_revision_service: Mock,
    fxt_project_service: Mock,
    fxt_db_session_factory: Callable,
) -> ExportDataset:
    fxt_project_service.get_project_by_id.return_value.name = "my_project"
    return ExportDataset(
        staged_datasets_dir=fxt_staged_datasets_dir,
        dataset_service=fxt_dataset_service,
        dataset_revision_service=fxt_dataset_revision_service,
        project_service=fxt_project_service,
        db_session_factory=fxt_db_session_factory,
    )


@pytest.fixture
def fxt_export_params() -> ExportDatasetJobParams:
    return ExportDatasetJobParams(
        project_id=uuid4(),
        task=Task(task_type=TaskType.DETECTION),
        export_format=DatasetFormat.COCO,
        labels=["label1", "label2"],
        subsets=[DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION],
        include_unannotated=False,
    )


class TestDatasetExporter:
    @pytest.mark.parametrize(
        "include_unannotated, subsets, labels",
        [
            (True, [DatasetItemSubset.TESTING], ["label1"]),
            (False, [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION], ["label1", "label2"]),
            (True, None, None),
        ],
    )
    def test_prepare_project_dataset(
        self,
        include_unannotated: bool,
        subsets: list[DatasetItemSubset] | None,
        labels: list[str] | None,
        fxt_export: ExportDataset,
        fxt_dataset_service: Mock,
        fxt_export_params: ExportDatasetJobParams,
    ):
        dataset = MagicMock(spec=Dataset)
        dataset.__len__.return_value = 10
        dataset.filter_by_subset.return_value = dataset
        dataset.filter_by_labels.return_value = dataset
        fxt_dataset_service.get_dm_dataset.return_value = dataset
        fxt_export_params.include_unannotated = include_unannotated
        fxt_export_params.subsets = subsets
        fxt_export_params.labels = labels

        fxt_export.prepare_dataset(fxt_export_params)

        fxt_dataset_service.get_dm_dataset.assert_called_once_with(
            project_id=fxt_export_params.project_id,
            task=fxt_export_params.task,
            annotation_status=None if include_unannotated else DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            sample_mode=SampleMode.IMPORT_EXPORT,
        )
        if subsets:
            dataset.filter_by_subset.assert_called_once_with(subset=[Subset[subset.name] for subset in subsets])
        if labels:
            dataset.filter_by_labels.assert_called_once_with(
                labels=labels, keep_empty_samples=fxt_export_params.include_unannotated
            )

    @pytest.mark.parametrize(
        "subsets, labels",
        [
            ([DatasetItemSubset.TESTING], ["label1"]),
            ([DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION], ["label1", "label2"]),
            (None, None),
        ],
    )
    def test_prepare_dataset_revision(
        self,
        subsets: list[DatasetItemSubset] | None,
        labels: list[str] | None,
        fxt_export: ExportDataset,
        fxt_dataset_revision_service: Mock,
        fxt_export_params: ExportDatasetJobParams,
    ):
        dataset = MagicMock(spec=Dataset)
        dataset.__len__.return_value = 10
        dataset.filter_by_subset.return_value = dataset
        dataset.filter_by_labels.return_value = dataset
        fxt_dataset_revision_service.load_revision.return_value = dataset
        fxt_export_params.dataset_id = uuid4()
        fxt_export_params.subsets = subsets
        fxt_export_params.labels = labels

        dataset_id, _ = fxt_export.prepare_dataset(fxt_export_params)

        assert dataset_id != fxt_export_params.dataset_id
        fxt_dataset_revision_service.load_revision.assert_called_once_with(
            project_id=fxt_export_params.project_id,
            dataset_revision_id=fxt_export_params.dataset_id,
        )
        if subsets:
            dataset.filter_by_subset.assert_called_once_with(subset=[Subset[subset.name] for subset in subsets])
        if labels:
            dataset.filter_by_labels.assert_called_once_with(
                labels=labels, keep_empty_samples=fxt_export_params.include_unannotated
            )

    @pytest.mark.parametrize(
        "export_format, data_format",
        [
            (DatasetFormat.COCO, DataFormat.COCO),
            (DatasetFormat.YOLO, DataFormat.YOLO),
        ],
    )
    def test_export_dataset(
        self,
        export_format: DatasetFormat,
        data_format: DataFormat,
        fxt_export: ExportDataset,
        fxt_staged_datasets_dir: Path,
    ):
        dataset = MagicMock(spec=Dataset)
        dataset_id = uuid4()
        project_name = "my_project"

        with patch("app.execution.dataset_export.export.export_dataset") as mock_export_dataset:
            target_dir = fxt_export.export_dataset(dataset_id, dataset, export_format, project_name)

            assert target_dir
            assert target_dir == fxt_staged_datasets_dir / str(dataset_id)
            mock_export_dataset.assert_called_once_with(
                dataset=dataset,
                data_format=data_format,
                output_path=str(fxt_staged_datasets_dir / str(dataset_id) / f"my_project-{export_format}-dataset.zip"),
                as_zip=True,
            )

    def test_export_dataset_geti(self, fxt_export: ExportDataset, fxt_staged_datasets_dir: Path):
        dataset = MagicMock(spec=Dataset)
        dataset_id = uuid4()
        project_name = "my_project"

        with (
            patch("app.execution.dataset_export.export.export_dataset") as mock_export_dataset,
        ):
            target_dir = fxt_export.export_dataset(dataset_id, dataset, DatasetFormat.GETI, project_name)

            assert target_dir
            assert target_dir == fxt_staged_datasets_dir / str(dataset_id)
            mock_export_dataset.assert_called_once_with(
                dataset=dataset, output_path=str(target_dir / f"my_project-{DatasetFormat.GETI}-dataset.zip"), as_zip=True
            )

    def test_execute(self, fxt_export: ExportDataset, fxt_export_params: ExportDatasetJobParams):
        dataset_id = uuid4()
        dataset = MagicMock(spec=Dataset)
        dataset.__len__.return_value = 10

        with (
            patch.object(fxt_export, "prepare_dataset", return_value=(dataset_id, dataset)) as mock_prepare,
            patch.object(fxt_export, "export_dataset") as mock_export,
            patch.object(fxt_export, "update_metadata") as mock_update_metadata,
        ):
            fxt_export.execute(fxt_export_params)

            mock_prepare.assert_called_once_with(fxt_export_params)
            mock_update_metadata.assert_called_once_with({"dataset_id": dataset_id})
            mock_export.assert_called_once_with(dataset_id, dataset, fxt_export_params.export_format, "my_project")

    def test_execute_empty_dataset(self, fxt_export: ExportDataset, fxt_export_params: ExportDatasetJobParams):
        dataset_id = uuid4()
        dataset = MagicMock(spec=Dataset)
        dataset.__len__.return_value = 0

        with (
            patch.object(fxt_export, "prepare_dataset", return_value=(dataset_id, dataset)) as mock_prepare,
            patch.object(fxt_export, "export_dataset") as mock_export,
            patch.object(fxt_export, "pin_message") as mock_pin_message,
        ):
            fxt_export.execute(fxt_export_params)

            mock_prepare.assert_called_once_with(fxt_export_params)
            mock_pin_message.assert_called_once_with("Dataset is empty after applying filters. Nothing to export.")
            mock_export.assert_not_called()
