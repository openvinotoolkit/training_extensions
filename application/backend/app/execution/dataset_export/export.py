# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import UUID, uuid4

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.base import DataFormat
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import Subset
from loguru import logger
from pathvalidate import sanitize_filename
from sqlalchemy.orm import Session

from app.datumaro_converter import SampleMode
from app.execution.base import Execution, step
from app.models import DatasetFormat, DatasetItemAnnotationStatus, ExportDatasetJobParams
from app.services import DatasetRevisionService, DatasetService, ProjectService


def get_dm_format(dataset_format: DatasetFormat) -> DataFormat:
    """
    Convert DatasetFormat to Datumaro DataFormat.

    Args:
        dataset_format: DatasetFormat enum value.

    Returns:
        Corresponding DataFormat enum value.
    """
    format_mapping = {
        DatasetFormat.COCO: DataFormat.COCO,
        DatasetFormat.YOLO: DataFormat.YOLO,
        DatasetFormat.VOC: DataFormat.VOC,
    }
    if dataset_format not in format_mapping:
        raise ValueError(f"Unsupported dataset format for export: {dataset_format}")
    return format_mapping[dataset_format]


class ExportDataset(Execution[ExportDatasetJobParams]):
    params_type = ExportDatasetJobParams

    def __init__(
        self,
        staged_datasets_dir: Path,
        dataset_service: DatasetService,
        dataset_revision_service: DatasetRevisionService,
        project_service: ProjectService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ):
        super().__init__()
        self._staged_datasets_dir = staged_datasets_dir
        self._dataset_service = dataset_service
        self._dataset_revision_service = dataset_revision_service
        self._project_service = project_service
        self._db_session_factory = db_session_factory

    @step("Prepare dataset for export", 20)
    def prepare_dataset(self, export_params: ExportDatasetJobParams) -> tuple[UUID, Dataset]:
        with self._db_session_factory() as session:
            if export_params.dataset_id is None:
                self._dataset_service.set_db_session(session)
                annotation_status = (
                    None if export_params.include_unannotated else DatasetItemAnnotationStatus.WITH_ANNOTATIONS
                )
                dataset = self._dataset_service.get_dm_dataset(
                    project_id=export_params.project_id,
                    task=export_params.task,
                    annotation_status=annotation_status,
                    sample_mode=SampleMode.IMPORT_EXPORT,
                )
            else:
                self._dataset_revision_service.set_db_session(session)
                dataset = self._dataset_revision_service.load_revision(
                    project_id=export_params.project_id, dataset_revision_id=export_params.dataset_id
                )
            if len(dataset) > 0 and export_params.subsets:
                dataset = dataset.filter_by_subset(subset=[Subset[subset.name] for subset in export_params.subsets])
            if len(dataset) > 0 and export_params.labels:
                dataset = dataset.filter_by_labels(
                    labels=export_params.labels, keep_empty_samples=export_params.include_unannotated
                )
            return uuid4(), dataset

    @step("Export dataset", 100)
    def export_dataset(
        self, dataset_id: UUID, dataset: Dataset, export_format: DatasetFormat, project_name: str
    ) -> Path | None:
        target_dir = self._staged_datasets_dir / str(dataset_id)
        logger.info("Exporting dataset {} to {} in {} format", dataset_id, target_dir, export_format)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = sanitize_filename(project_name)[:32]
        filename = f"{file_prefix}-{export_format}-dataset.zip"
        match export_format:
            case DatasetFormat.COCO | DatasetFormat.YOLO | DatasetFormat.VOC:
                export_dataset(
                    dataset=dataset,
                    data_format=get_dm_format(export_format),
                    output_path=str(target_dir / filename),
                    as_zip=True,
                )
            case DatasetFormat.GETI:
                export_dataset(
                    dataset=dataset,
                    output_path=str(target_dir / filename),
                    as_zip=True,
                )
            case _:
                raise ValueError(f"Unsupported dataset format for export: {export_format}")
        return target_dir

    def execute(self, params: ExportDatasetJobParams) -> None:
        dataset_id, dataset = self.prepare_dataset(params)
        if len(dataset) == 0:
            self.pin_message("Dataset is empty after applying filters. Nothing to export.")
            return
        self.update_metadata({"dataset_id": dataset_id})
        with self._db_session_factory() as session:
            self._project_service.set_db_session(session)
            project = self._project_service.get_project_by_id(params.project_id)
            project_name = project.name
        self.export_dataset(dataset_id, dataset, params.export_format, project_name)
