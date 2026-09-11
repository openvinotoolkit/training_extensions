# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.schema import DatasetViewDB
from app.models.dataset import DatasetStatistics
from app.models.dataset_item import DatasetItem
from app.models.dataset_view import DatasetView
from app.models.media import Media, MediaAdapter, MediaType, Video
from app.repositories import DatasetViewRepository, MediaRepository
from app.repositories.base import UniqueConstraintIntegrityError
from app.services.dataset_service import DatasetItemFilters
from app.services.media_service import MediaFilters

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType, ResourceWithNameAlreadyExistsError


class DatasetViewService(BaseSessionManagedService):
    """
    Service for managing dataset views: named, user-defined subsets of a project's dataset media.

    Design notes:
        - Membership (which media belongs to which view) is tracked at the *media* level (images and videos),
          not at the dataset-item level. This allows a video to be assigned to a view immediately upon upload,
          even before any of its frames have been annotated (and therefore before a dataset item exists for it).
        - Media deletion is always a dataset-wide (not view-scoped) operation, handled by ``MediaService`` /
          the existing ``/dataset/media`` endpoints. Deleting a media item removes it from the project's
          dataset entirely, and it is automatically removed from any view it was assigned to (cascading delete).
          Unassigning a media item from a view, on the other hand, only removes it from that view; the media
          itself, and its membership in other views, is unaffected.
        - Uploading media while a specific view is selected is a two-step client-side operation (handled by the
          UI): (1) upload the media via the regular ``/dataset/media`` endpoint, then (2) assign it to the
          selected view via :meth:`assign_media`.

    NOTE: This is a work-in-progress feature. The API surface (routers, schemas, dependencies) is fully wired up,
    but the business logic is not implemented yet; every method below raises ``NotImplementedError``.
    """

    def __init__(self, db_session: Session | None = None) -> None:
        super().__init__(db_session)

    def _get_repo(self, project_id: UUID) -> DatasetViewRepository:
        return DatasetViewRepository(project_id=str(project_id), db=self.db_session)

    @staticmethod
    def _get_db_dataset_view(repo: DatasetViewRepository, dataset_view_id: UUID) -> DatasetViewDB:
        db_dataset_view = repo.get_by_id(str(dataset_view_id))
        if db_dataset_view is None:
            raise ResourceNotFoundError(ResourceType.DATASET_VIEW, str(dataset_view_id))
        return db_dataset_view

    def _validate_media_exist(self, project_id: UUID, media_ids: list[UUID]) -> None:
        """
        Raise ResourceNotFoundError if any of the given media ids does not exist in the project, or if it
        refers to a video frame.

        Dataset view membership is only tracked at the image/video level (see class docstring); video frames
        are considered part of a view solely via their parent video. Video frame ids are therefore treated as
        non-existent in this context, to keep view membership consistent across all of the service's read
        paths (media listing/counting explicitly exclude frames, and items/statistics derive frame membership
        from the parent video rather than from any direct frame assignment).
        """
        media_repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        found_media = {
            db_media.id: db_media.type for db_media in media_repo.get_by_ids([str(media_id) for media_id in media_ids])
        }
        for media_id in media_ids:
            media_type = found_media.get(str(media_id))
            if media_type is None or media_type == MediaType.VIDEO_FRAME:
                raise ResourceNotFoundError(ResourceType.MEDIA, str(media_id))

    def create_dataset_view(self, project_id: UUID, name: str, media_ids: list[UUID] | None = None) -> DatasetView:
        """Create a new, named dataset view, optionally pre-populated with the given media items."""
        repo = self._get_repo(project_id)
        db_dataset_view = DatasetViewDB(project_id=str(project_id), name=name)
        try:
            db_dataset_view = repo.save(db_dataset_view)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.DATASET_VIEW, name)

        if media_ids:
            self._validate_media_exist(project_id, media_ids)
            repo.assign_media(dataset_view_id=db_dataset_view.id, media_ids=[str(media_id) for media_id in media_ids])

        return DatasetView.model_validate(db_dataset_view)

    def list_dataset_views(self, project_id: UUID) -> list[DatasetView]:
        """List the dataset views belonging to a project."""
        repo = self._get_repo(project_id)
        return [DatasetView.model_validate(db_dataset_view) for db_dataset_view in repo.list_by_project()]

    def get_dataset_view_by_id(self, project_id: UUID, dataset_view_id: UUID) -> DatasetView:
        """Get a dataset view by its ID."""
        repo = self._get_repo(project_id)
        db_dataset_view = self._get_db_dataset_view(repo, dataset_view_id)
        return DatasetView.model_validate(db_dataset_view)

    def rename_dataset_view(self, project_id: UUID, dataset_view_id: UUID, new_name: str) -> DatasetView:
        """Rename a dataset view."""
        repo = self._get_repo(project_id)
        db_dataset_view = self._get_db_dataset_view(repo, dataset_view_id)

        if new_name != db_dataset_view.name:
            db_dataset_view.name = new_name
            try:
                db_dataset_view = repo.save(db_dataset_view)
            except UniqueConstraintIntegrityError:
                raise ResourceWithNameAlreadyExistsError(ResourceType.DATASET_VIEW, new_name)

        return DatasetView.model_validate(db_dataset_view)

    def delete_dataset_view(self, project_id: UUID, dataset_view_id: UUID) -> None:
        """Delete a dataset view. This does not affect the underlying media or dataset items."""
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        repo.delete(str(dataset_view_id))

    def assign_media(self, project_id: UUID, dataset_view_id: UUID, media_ids: list[UUID]) -> None:
        """
        Assign one or more media items (images or videos) to a dataset view, in bulk.

        Assigning a media item that is already part of the view is a no-op for that item. A media item may be
        assigned to multiple views at the same time.
        """
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        if media_ids:
            self._validate_media_exist(project_id, media_ids)
            repo.assign_media(dataset_view_id=str(dataset_view_id), media_ids=[str(media_id) for media_id in media_ids])

    def unassign_media(self, project_id: UUID, dataset_view_id: UUID, media_ids: list[UUID]) -> None:
        """
        Unassign one or more media items from a dataset view, in bulk.

        This only removes the media items from the given view; the media itself is not deleted and remains
        part of the project's dataset (and any other view it is assigned to).
        """
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        repo.unassign_media(dataset_view_id=str(dataset_view_id), media_ids=[str(media_id) for media_id in media_ids])

    def count_dataset_view_media(
        self, project_id: UUID, dataset_view_id: UUID, filters: MediaFilters | None = None
    ) -> int:
        """Count the media items (images and videos) assigned to a dataset view, optionally matching filters."""
        if filters is None:
            filters = MediaFilters()
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        return repo.count_media(
            dataset_view_id=str(dataset_view_id),
            start_date=filters.start_date,
            end_date=filters.end_date,
            annotation_status=filters.annotation_status,
            label_ids=label_ids_str,
            subsets=filters.subsets,
        )

    def list_dataset_view_media(
        self, project_id: UUID, dataset_view_id: UUID, filters: MediaFilters | None = None
    ) -> list[Media]:
        """List (filter) the media items (images and videos) assigned to a dataset view."""
        if filters is None:
            filters = MediaFilters()
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        media_dbs = repo.list_media(
            dataset_view_id=str(dataset_view_id),
            limit=filters.limit,
            offset=filters.offset,
            start_date=filters.start_date,
            end_date=filters.end_date,
            annotation_status=filters.annotation_status,
            label_ids=label_ids_str,
            subsets=filters.subsets,
            sort_by=filters.sort_by,
            sort_direction=filters.sort_direction,
        )
        media_repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        return [
            Video.model_validate(media_db).model_copy(
                update={"annotated_frame_count": media_repo.count_annotated_video_frames_by_video_id(media_db.id)}
            )
            if media_db.type == MediaType.VIDEO
            else MediaAdapter.validate_python(media_db)
            for media_db in media_dbs
        ]

    def list_dataset_view_media_ids(
        self, project_id: UUID, dataset_view_id: UUID, filters: MediaFilters | None = None
    ) -> tuple[tuple[UUID, MediaType], ...]:
        """Get the id and type of every media item assigned to a dataset view, without loading full media rows."""
        if filters is None:
            filters = MediaFilters()
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        media_id_rows = repo.list_media_ids(
            dataset_view_id=str(dataset_view_id),
            start_date=filters.start_date,
            end_date=filters.end_date,
            annotation_status=filters.annotation_status,
            label_ids=label_ids_str,
            subsets=filters.subsets,
        )
        return tuple((UUID(media_id), MediaType(media_type)) for media_id, media_type in media_id_rows)

    def count_dataset_view_items(
        self, project_id: UUID, dataset_view_id: UUID, filters: DatasetItemFilters | None = None
    ) -> int:
        """Count the dataset items assigned to a dataset view, optionally matching the given filters."""
        if filters is None:
            filters = DatasetItemFilters()
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        return repo.count_items(
            dataset_view_id=str(dataset_view_id),
            start_date=filters.start_date,
            end_date=filters.end_date,
            annotation_status=filters.annotation_status,
            label_ids=label_ids_str,
            subsets=filters.subsets,
        )

    def list_dataset_view_items(
        self, project_id: UUID, dataset_view_id: UUID, filters: DatasetItemFilters | None = None
    ) -> list[DatasetItem]:
        """List (filter) the dataset items assigned to a dataset view."""
        if filters is None:
            filters = DatasetItemFilters()
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        db_dataset_items = repo.list_items(
            dataset_view_id=str(dataset_view_id),
            limit=filters.limit,
            offset=filters.offset,
            start_date=filters.start_date,
            end_date=filters.end_date,
            annotation_status=filters.annotation_status,
            label_ids=label_ids_str,
            subsets=filters.subsets,
            sort_by=filters.sort_by,
            sort_direction=filters.sort_direction,
        )
        return [DatasetItem.model_validate(db_dataset_item) for db_dataset_item in db_dataset_items]

    def get_dataset_view_statistics(self, project_id: UUID, dataset_view_id: UUID) -> DatasetStatistics:
        """Get statistics (media & annotation counts) about the items assigned to a dataset view."""
        repo = self._get_repo(project_id)
        self._get_db_dataset_view(repo, dataset_view_id)
        statistics_dict = repo.get_statistics(str(dataset_view_id))
        return DatasetStatistics.model_validate(statistics_dict)
