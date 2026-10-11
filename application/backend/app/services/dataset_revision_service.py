# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import polars as pl
from datumaro.experimental import Dataset
from loguru import logger
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import DatasetRevisionDB
from app.models import DatasetItemSubset
from app.models.dataset_item_revision import DatasetRevisionItem
from app.models.dataset_revision import DatasetRevision, DatasetRevisionCounts
from app.models.media import ImageFormat
from app.repositories import DatasetRevisionRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError
from app.utils.images import convert_to_jpeg_compatible, crop_to_thumbnail

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType
from .media_service import InvalidImageError

# Thumbnails for dataset revisions are generated on the fly and need to be smaller than pregenerated thumbnails
DATASET_REVISION_ITEM_THUMBNAIL_SIZE = 128


class DatasetRevisionService(BaseSessionManagedService):
    def __init__(self, data_dir: Path, db_session: Session | None = None) -> None:
        super().__init__(db_session)
        self.projects_dir = data_dir / "projects"

    def save_revision(
        self,
        project_id: UUID,
        dataset: Dataset,
        dataset_view_id: UUID | None = None,
        dataset_view_name: str | None = None,
    ) -> UUID:
        """
        Saves the dataset as a new revision.

        Creates a new dataset revision entry in the database and exports the dataset
        to a zip file in the project's revisions directory.

        Args:
            project_id: The UUID of the project to save the revision for.
            dataset: The Datumaro dataset to export.
            dataset_view_id: The UUID of the dataset view the revision was built from, if any.
            dataset_view_name: The name of that dataset view, used to name the revision.

        Returns:
            UUID: The UUID of the newly created dataset revision.
        """
        from datumaro.experimental.export_import import export_dataset

        item_counts = self._count_dataset_revision_items(dataset=dataset)
        if not (item_counts.training and item_counts.validation and item_counts.testing):
            scope = f"dataset view '{dataset_view_name or dataset_view_id}'" if dataset_view_id else "dataset"
            raise ValueError(
                f"Cannot save dataset revision for {project_id}: the {scope} does not contain enough annotated "
                f"items to fill the training, validation and testing subsets. Item counts: {item_counts}"
            )
        revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        dataset_revision_id = str(uuid4())
        short_id = dataset_revision_id.split("-")[0]
        dataset_name = f"{dataset_view_name} ({short_id})" if dataset_view_name else f"Dataset ({short_id})"

        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / dataset_revision_id
        logger.info("Saving dataset revision '{}' to '{}'.", dataset_revision_id, revision_path)
        export_dataset(
            dataset=dataset,
            output_path=revision_path,
            as_zip=False,  # Export as uncompressed directory, see #5070 for details
        )
        size_in_bytes = sum(item.stat().st_size for item in revision_path.rglob("*") if item.is_file())

        try:
            revision_db = revision_repo.save(
                DatasetRevisionDB(
                    id=dataset_revision_id,
                    project_id=str(project_id),
                    name=dataset_name,
                    dataset_view_id=str(dataset_view_id) if dataset_view_id is not None else None,
                    total_count=item_counts.total,
                    training_count=item_counts.training,
                    validation_count=item_counts.validation,
                    testing_count=item_counts.testing,
                    size=size_in_bytes,
                    created_at=datetime.now(UTC),  # Set created_at explicitly with microsecond precision.
                )
            )
        except (IntegrityError, PrimaryKeyIntegrityError, UniqueConstraintIntegrityError):
            logger.error("Could not save dataset revision '{}' in the database.", dataset_revision_id)
            if revision_path.exists():
                shutil.rmtree(revision_path)
                logger.info("Deleted dataset revision files at '{}'.", revision_path)
            raise

        return UUID(revision_db.id)

    def load_revision(self, project_id: UUID, dataset_revision_id: UUID) -> Dataset:
        """
        Loads the Datumaro dataset belonging to the dataset revision.

        Args:
            project_id: The UUID of the project.
            dataset_revision_id: The UUID of the dataset revision.
        Returns:
            Dataset: The dataset revision as a Datumaro dataset.
        """
        from datumaro.experimental.export_import import import_dataset

        dataset_revision = self.get_dataset_revision(project_id, dataset_revision_id)
        if dataset_revision.files_deleted:
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(dataset_revision_id))
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision_id)
        return import_dataset(input_path=parquet_path.parent)

    def get_latest_uptodate_dataset_revision(
        self, project_id: UUID, dataset_view_id: UUID | None = None
    ) -> DatasetRevision | None:
        """
        Get latest up to date created dataset revision in a project, if it exists.

        Up to date means the dataset revision was created after the last update on any dataset item it covers.

        Args:
            project_id (UUID): The UUID of the project
            dataset_view_id (UUID | None): If given, only consider revisions created from this dataset view;
                otherwise only consider revisions covering the entire dataset.

        Returns:
            DatasetRevision: The latest created dataset revision in a project or None if none exists
        """
        dataset_revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        dataset_revision_db = dataset_revision_repo.get_latest_uptodate_dataset_revision(
            dataset_view_id=str(dataset_view_id) if dataset_view_id is not None else None
        )
        if dataset_revision_db is None:
            return None
        return DatasetRevision.model_validate(dataset_revision_db)

    def list_dataset_revisions(self, project_id: UUID) -> list[DatasetRevision]:
        """
        Get information about all available dataset revisions in a project.

        Retrieves a list of all dataset revisions that belong to the specified project.

        Args:
            project_id (UUID): The unique identifier of the project whose dataset revisions to list.

        Returns:
            list[DatasetRevision]: A list of dataset revision objects representing all dataset
                revisions in the project. Returns an empty list if the project has no dataset revisions.
        """
        dataset_revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        return [DatasetRevision.model_validate(dataset_rev_db) for dataset_rev_db in dataset_revision_repo.list_all()]

    def get_dataset_revision(self, project_id: UUID, revision_id: UUID) -> DatasetRevision:
        """
        Get a dataset revision by ID.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Returns:
            DatasetRevision: The dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        revision = revision_repo.get_by_id(str(revision_id))
        if revision is None:
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))
        return self._to_dataset_revision(dataset_db=revision)

    def rename_dataset_revision(
        self, project_id: UUID, dataset_revision: DatasetRevision, new_name: str
    ) -> DatasetRevision:
        """
        Rename a dataset revision.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision to rename.
            new_name: The new name to assign to the dataset revision.

        Returns:
            DatasetRevision: The dataset revision object containing the dataset revision's updated information.
        """
        if new_name is not None:
            dataset_revision.name = new_name
            self.update_dataset_revision(project_id=project_id, dataset_revision=dataset_revision)
        return dataset_revision

    def update_dataset_revision(self, project_id: UUID, dataset_revision: DatasetRevision) -> None:
        """
        Updates a dataset revision.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision to update.
        """
        revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        _ = revision_repo.update(
            DatasetRevisionDB(
                id=str(dataset_revision.id),
                project_id=str(project_id),
                name=dataset_revision.name,
                dataset_view_id=str(dataset_revision.dataset_view_id)
                if dataset_revision.dataset_view_id is not None
                else None,
                files_deleted=dataset_revision.files_deleted,
                size=0 if dataset_revision.files_deleted else dataset_revision.size,
                total_count=dataset_revision.item_counts.total,
                training_count=dataset_revision.item_counts.training,
                validation_count=dataset_revision.item_counts.validation,
                testing_count=dataset_revision.item_counts.testing,
            )
        )

    def delete_dataset_revision(
        self,
        project_id: UUID,
        revision_id: UUID,
    ) -> None:
        """
        Deletes a DatasetRevision from the DB and deletes its associated files from the disk.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        try:
            revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
            revision_repo.delete(str(revision_id))
        except IntegrityError:
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))

        self._delete_dataset_revision_files(project_id=project_id, dataset_revision_id=revision_id)

    def delete_dataset_revision_files(
        self,
        project_id: UUID,
        revision_id: UUID,
    ) -> None:
        """
        Marks the DatasetRevision files as deleted, and deletes associated files from the disk.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        dataset_revision = self.get_dataset_revision(project_id=project_id, revision_id=revision_id)

        if not dataset_revision.files_deleted:
            dataset_revision.files_deleted = True
            self.update_dataset_revision(project_id=project_id, dataset_revision=dataset_revision)

        self._delete_dataset_revision_files(project_id=project_id, dataset_revision_id=revision_id)

    def _delete_dataset_revision_files(self, project_id: UUID, dataset_revision_id: UUID) -> None:
        """Deletes files associated with dataset revision from hard disk"""
        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(dataset_revision_id)
        if revision_path.exists():
            shutil.rmtree(revision_path)
            logger.info("Deleted dataset revision files at '{}'", revision_path)
        else:
            logger.info("Files for dataset revision '{}' are already deleted", dataset_revision_id)

    def _get_revision_parquet_path(self, project_id: UUID, revision_id: UUID) -> Path:
        """
        Get the path to the parquet file for a dataset revision.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Returns:
            Path to the data.parquet file.

        Raises:
            ResourceNotFoundError: If the parquet file is not found.
        """
        parquet_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(revision_id) / "data.parquet"
        if not parquet_path.exists():
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))

        return parquet_path

    def _count_dataset_revision_items(self, dataset: Dataset) -> DatasetRevisionCounts:
        """
        Count the number of dataset items in a dataset revision, grouped by subset.

        - The total number of items ("total")
        - The number of items in each subset (e.g., "training", "validation", "testing")

        Args:
            dataset: A Datumaro dataset.

        Returns:
            DatasetRevisionCounts: The counts of dataset items.
        """
        subset_counts_raw = (
            dataset.df.lazy().group_by("subset").len().collect().to_dicts()
        )  # [{"subset": "TRAINING", "len": 12}, ...]
        counts_by_subset = {row["subset"].lower(): row["len"] for row in subset_counts_raw}  # {"training": 12, ...}
        return DatasetRevisionCounts(
            training=counts_by_subset.get("training", 0),
            validation=counts_by_subset.get("validation", 0),
            testing=counts_by_subset.get("testing", 0),
            total=sum(counts_by_subset.values()),
        )

    def _parse_revision_item_from_df_row_dict(
        self, project_id: UUID, dataset_revision_id: UUID, df_revision_item: dict[str, Any]
    ) -> DatasetRevisionItem:
        """
        Utility method to convert a dataset revision item from a dict representation
        (a row of the Polars dataframe, serialized with to_dicts()) to a DatasetRevisionItem.

        The dict is expected to have the following structure:
        {
            "id": "9bedc1ec-edf1-4eec-8549-82239c585c48",
            "image": "relative/path/to/image.jpg",
            "image_info": {
                "width": 1024,
                "height": 768
            },
            "subset": "TRAINING",
            ... other fields ...
        }

        Args:
            project_id: The UUID of the project.
            dataset_revision_id: The UUID of the dataset revision.
            df_revision_item: The dataset revision item as a dict.

        Returns:
            DatasetRevisionItem
        """
        dataset_revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(dataset_revision_id)
        try:
            # The value stored in the 'image' column is expected to be the name of the image file,
            # relative to the dataset revision's images/ directory.
            image_path = dataset_revision_path / "images" / Path(df_revision_item["image"])

            return DatasetRevisionItem(
                id=UUID(df_revision_item["id"]),
                format=ImageFormat(image_path.suffix.lstrip(".").lower()),
                image_path=image_path,
                width=df_revision_item["image_info"]["width"],
                height=df_revision_item["image_info"]["height"],
                subset=DatasetItemSubset(df_revision_item["subset"].lower()),
            )
        except Exception as e:
            logger.error(
                "Failed to parse dataset revision item from dataframe row dict: {}. Error: {}",
                df_revision_item,
                e,
            )
            raise

    def list_dataset_revision_items(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        limit: int = 10,
        offset: int = 0,
        subsets: list[DatasetItemSubset] | None = None,
    ) -> tuple[list[DatasetRevisionItem], int]:
        """
        List items in a dataset revision with pagination and filtering.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            limit: Maximum number of items to return.
            offset: Number of items to skip.
            subsets: Optional list of subsets to filter by.

        Returns:
            Tuple of (list of items as dicts, total count).
        """
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision.id)

        df = pl.scan_parquet(parquet_path)

        if subsets:
            df = df.filter(pl.col("subset").is_in([s.name for s in subsets]))

        total_count = df.select(pl.len()).collect().item()
        df = df.slice(offset, limit).collect()

        dataset_revision_items = []
        for item in df.to_dicts():
            dataset_revision_items.append(
                self._parse_revision_item_from_df_row_dict(
                    project_id=project_id, dataset_revision_id=dataset_revision.id, df_revision_item=item
                )
            )
        return dataset_revision_items, total_count

    def get_dataset_revision_item(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        item_id: str,
    ) -> DatasetRevisionItem:
        """
        Get a specific item from a dataset revision by ID.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            DatasetRevisionItem: The requested dataset revision item.

        Raises:
            ResourceNotFoundError: If the item is not found.
        """
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision.id)

        df = pl.scan_parquet(parquet_path)

        if "id" in df.columns:
            df = df.filter(pl.col("id") == item_id).collect()
        else:  # Fallback: locate the item through the 'image' column, if it contains the item ID
            logger.warning("Dataset revision parquet does not contain 'id' column, falling back to search via 'image'")
            df = df.filter(pl.col("image").str.contains(item_id)).collect()
        if len(df) == 0:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, item_id)

        item = df.to_dicts()[0]
        item["id"] = item_id  # Ensure the ID is set correctly

        return self._parse_revision_item_from_df_row_dict(
            project_id=project_id, dataset_revision_id=dataset_revision.id, df_revision_item=item
        )

    def get_dataset_revision_item_thumbnail(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        item_id: str,
    ) -> Image.Image:
        """
        Generate and return a thumbnail for a specific dataset revision item.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            Image.Image: The generated thumbnail image.
        """
        binary_path = self.get_dataset_revision_item(
            project_id=project_id,
            dataset_revision=dataset_revision,
            item_id=item_id,
        ).image_path
        try:
            with Image.open(binary_path) as image:
                thumbnail = crop_to_thumbnail(
                    image=image,
                    target_width=DATASET_REVISION_ITEM_THUMBNAIL_SIZE,
                    target_height=DATASET_REVISION_ITEM_THUMBNAIL_SIZE,
                )
            # Ensure thumbnail is in a JPEG-compatible mode before it is encoded downstream.
            # High bit depth images (e.g. 16-bit) are normalized so they are not washed out;
            # a plain .convert("RGB") would keep only the high byte and produce white thumbnails.
            thumbnail = convert_to_jpeg_compatible(thumbnail)
        except UnidentifiedImageError:
            logger.error("Failed to open image {} for thumbnail generation", binary_path)
            raise InvalidImageError("Failed to open image for thumbnail generation.")
        return thumbnail

    @staticmethod
    def _to_dataset_revision(dataset_db: DatasetRevisionDB) -> DatasetRevision:
        """Convert database model to DatasetRevision."""
        return DatasetRevision.model_validate(dataset_db, from_attributes=True)
