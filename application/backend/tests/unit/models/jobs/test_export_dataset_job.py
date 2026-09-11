# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models import DatasetFormat, ExportDatasetJobParams, Task, TaskType


class TestExportDatasetJobParams:
    @pytest.mark.parametrize(
        ("task_type", "export_format"),
        [
            (TaskType.CLASSIFICATION, DatasetFormat.GETI),
            (TaskType.CLASSIFICATION, DatasetFormat.VOC),
            (TaskType.DETECTION, DatasetFormat.GETI),
            (TaskType.DETECTION, DatasetFormat.COCO),
            (TaskType.DETECTION, DatasetFormat.YOLO),
            (TaskType.INSTANCE_SEGMENTATION, DatasetFormat.GETI),
            (TaskType.INSTANCE_SEGMENTATION, DatasetFormat.COCO),
        ],
    )
    def test_validate_format_valid(self, task_type: TaskType, export_format: DatasetFormat):
        """Test that valid combinations of task and export format are accepted."""
        try:
            ExportDatasetJobParams(
                project_id=uuid4(),
                task=Task(task_type=task_type),
                export_format=export_format,
            )
        except ValidationError:
            pytest.fail("Validation failed for a valid combination.")

    @pytest.mark.parametrize(
        ("task_type", "export_format"),
        [
            (TaskType.CLASSIFICATION, DatasetFormat.COCO),
            (TaskType.CLASSIFICATION, DatasetFormat.YOLO),
            (TaskType.DETECTION, DatasetFormat.VOC),
            (TaskType.INSTANCE_SEGMENTATION, DatasetFormat.YOLO),
            (TaskType.INSTANCE_SEGMENTATION, DatasetFormat.VOC),
        ],
    )
    def test_validate_format_invalid(self, task_type: TaskType, export_format: DatasetFormat):
        """Test that invalid combinations of task and export format raise a ValueError."""
        with pytest.raises(
            ValidationError, match=f"Export format '{export_format}' is not supported for {task_type} task."
        ):
            ExportDatasetJobParams(
                project_id=uuid4(),
                task=Task(task_type=task_type),
                export_format=export_format,
            )

    def test_dataset_id_and_dataset_view_id_mutually_exclusive(self):
        """Test that setting both dataset_id and dataset_view_id raises a ValueError."""
        with pytest.raises(ValidationError, match="dataset_id and dataset_view_id are mutually exclusive"):
            ExportDatasetJobParams(
                project_id=uuid4(),
                task=Task(task_type=TaskType.DETECTION),
                export_format=DatasetFormat.COCO,
                dataset_id=uuid4(),
                dataset_view_id=uuid4(),
            )

    def test_dataset_view_id_alone_is_valid(self):
        """Test that dataset_view_id can be set without dataset_id."""
        params = ExportDatasetJobParams(
            project_id=uuid4(),
            task=Task(task_type=TaskType.DETECTION),
            export_format=DatasetFormat.COCO,
            dataset_view_id=uuid4(),
        )
        assert params.dataset_id is None
        assert params.dataset_view_id is not None
