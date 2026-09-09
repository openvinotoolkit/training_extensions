# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import os.path
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import numpy as np
from loguru import logger
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.db.schema import MediaDB
from app.models import DatasetItem, DatasetItemAnnotationStatus, Media, MediaType, Project, Video, VideoFrame
from app.models.media import ImageFormat, MediaAdapter, MediaSortBy, SortDirection, VideoFormat
from app.repositories import MediaRepository
from app.services.video import VideoService
from app.utils.images import convert_to_jpeg_compatible, crop_to_thumbnail

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType

DEFAULT_THUMBNAIL_SIZE = 256

VIDEO_WRITE_CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB


class InvalidImageError(Exception):
    """Exception raised when invalid image is used to create a media."""

    def __init__(self, message: str | None = None):
        msg = message or "Invalid image has been passed while creating a media."
        super().__init__(msg)


@dataclass(frozen=True)
class MediaFilters:
    limit: int = 20
    offset: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None
    annotation_status: DatasetItemAnnotationStatus | None = None
    label_ids: list[UUID] | None = None
    subsets: list[str] | None = None
    sort_by: MediaSortBy = MediaSortBy.UPLOAD_DATE
    sort_direction: SortDirection = SortDirection.DESC


@dataclass(frozen=True)
class ImageMetadata:
    project_id: UUID
    name: str
    image_format: ImageFormat
    data: Image.Image | np.ndarray | BinaryIO | BytesIO
    media_type: MediaType = MediaType.IMAGE
    source_id: UUID | None = None
    video_id: UUID | None = None
    frame_idx: int | None = None

    def __post_init__(self) -> None:
        if self.media_type == MediaType.VIDEO_FRAME:
            if self.video_id is None:
                raise ValueError("video_id must be provided when media_type is VIDEO_FRAME")
            if self.frame_idx is None:
                raise ValueError("frame_idx must be provided when media_type is VIDEO_FRAME")


class MediaService(BaseSessionManagedService):
    def __init__(
        self,
        data_dir: Path,
        video_service: VideoService | None = None,
        db_session: Session | None = None,
    ) -> None:
        super().__init__(db_session)
        self.projects_dir = data_dir / "projects"
        self._video_service = video_service

    def _get_video_service(self) -> VideoService:
        if self._video_service is None:
            # FIXME: direct instance as a tmp workaround to avoid sending non-pickable objects between process
            #  boundaries in the current job execution implementation.
            #  Should be refactored with a proper DI in lifecycle.py.
            self._video_service = VideoService()
        return self._video_service

    @staticmethod
    def _read_image_from_ndarray(data: np.ndarray) -> Image.Image:
        # Pillow does not support multi-channel uint16 arrays (e.g. 16-bit RGB from datumaro).
        # Such arrays are always grayscale duplicated across channels - collapse to single channel.
        if data.dtype == np.uint16 and data.ndim == 3 and data.shape[-1] in (1, 3, 4):
            if data.shape[-1] == 1 or (
                np.array_equal(data[:, :, 0], data[:, :, 1]) and np.array_equal(data[:, :, 0], data[:, :, 2])
            ):
                data = data[:, :, 0]
            else:
                raise InvalidImageError(
                    "Unsupported 16-bit multi-channel image array. Expected duplicated grayscale channels."
                )
        return Image.fromarray(data)

    @staticmethod
    def _read_image_from_binary(data: BinaryIO | BytesIO) -> Image.Image:
        data.seek(0)
        try:
            return Image.open(data)
        except UnidentifiedImageError:
            raise InvalidImageError

    @staticmethod
    def _generate_and_save_thumbnail(image: Image.Image, path: Path) -> None:
        try:
            thumbnail_image = crop_to_thumbnail(
                image=image, target_width=DEFAULT_THUMBNAIL_SIZE, target_height=DEFAULT_THUMBNAIL_SIZE
            )
            thumbnail_image = convert_to_jpeg_compatible(thumbnail_image)
            thumbnail_image.save(path)
        except Exception:
            logger.exception("Failed to generate thumbnail image")

    @staticmethod
    def _copy_binary(data: BinaryIO | BytesIO, path: Path, chunk_size: int = 1024 * 1024) -> None:
        data.seek(0)
        with path.open("wb") as output:
            while chunk := data.read(chunk_size):
                output.write(chunk)

    def create_image(
        self,
        metadata: ImageMetadata,
    ) -> Media:
        """Creates a new media (image)"""
        media_id = uuid4()
        original_binary: BinaryIO | BytesIO | None = None
        match metadata.data:
            case Image.Image():
                image = metadata.data
            case np.ndarray():
                image = self._read_image_from_ndarray(metadata.data)
            case _:
                original_binary = metadata.data
                image = self._read_image_from_binary(metadata.data)

        dataset_dir = self.projects_dir / f"{metadata.project_id}/dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        binary_path = dataset_dir / f"{media_id}.{metadata.image_format}"

        if original_binary is not None:
            self._copy_binary(original_binary, binary_path)
        else:
            try:
                image.save(binary_path, exif=image.getexif())
            except RuntimeError:
                logger.warning(
                    "Failed to save image with EXIF data ({}), saving without EXIF.",
                    metadata.name,
                )
                image.save(binary_path)

        try:
            MediaService._generate_and_save_thumbnail(image, dataset_dir / f"{media_id}-thumb.jpg")

            media = MediaDB(
                id=str(media_id),
                project_id=str(metadata.project_id),
                type=metadata.media_type,
                name=metadata.name,
                format=str(metadata.image_format),
                width=image.width,
                height=image.height,
                size=os.path.getsize(binary_path),
                source_id=str(metadata.source_id) if metadata.source_id is not None else None,
                video_id=str(metadata.video_id) if metadata.video_id is not None else None,
                frame_index=metadata.frame_idx,
            )

            repo = MediaRepository(project_id=str(metadata.project_id), db=self.db_session)
            db_media = repo.save(media)
        except Exception as e:
            binary_path.unlink(missing_ok=True)
            raise e
        return MediaAdapter.validate_python(db_media)

    def create_video(
        self,
        project_id: UUID,
        name: str,
        video_format: VideoFormat,
        data: BinaryIO,
        source_id: UUID | None = None,
    ) -> Media:
        """Creates a new media (video)"""
        media_id = uuid4()

        dataset_dir = self.projects_dir / f"{project_id}/dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        binary_path = dataset_dir / f"{media_id}.{video_format}"

        self._copy_binary(data, binary_path, VIDEO_WRITE_CHUNK_SIZE)

        try:
            video_metadata = self._get_video_service().get_video_metadata(video_path=binary_path)
            media = MediaDB(
                id=str(media_id),
                project_id=str(project_id),
                type=MediaType.VIDEO,
                name=name,
                format=str(video_format),
                width=video_metadata.width,
                height=video_metadata.height,
                frame_count=video_metadata.frame_count,
                fps=video_metadata.fps,
                size=os.path.getsize(binary_path),
                source_id=str(source_id) if source_id is not None else None,
            )

            video_frame = self._get_frame_binary_from_video_file(
                video_path=binary_path, frame_index=video_metadata.frame_count // 2
            )
            MediaService._generate_and_save_thumbnail(image=video_frame, path=dataset_dir / f"{media_id}-thumb.jpg")

            repo = MediaRepository(project_id=str(project_id), db=self.db_session)
            db_media = repo.save(media)
        except Exception as e:
            binary_path.unlink(missing_ok=True)
            raise e
        return MediaAdapter.validate_python(db_media)

    def count_media(
        self,
        project: Project,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: DatasetItemAnnotationStatus | None = None,
        label_ids: list[UUID] | None = None,
        subsets: list[str] | None = None,
        exclude_types: list[MediaType] | None = None,
    ) -> int:
        """Get number of available media (within date range if specified)"""
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in label_ids] if label_ids else None
        return repo.count(
            start_date=start_date,
            end_date=end_date,
            annotation_status=annotation_status,
            label_ids=label_ids_str,
            subsets=subsets,
            exclude_types=exclude_types,
        )

    def list_media(
        self,
        project_id: UUID,
        filters: MediaFilters | None = None,
        exclude_types: list[MediaType] | None = None,
    ) -> list[Media]:
        """Get information about available media"""
        if filters is None:
            filters = MediaFilters()
        repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        media_dbs = repo.list_items(
            limit=filters.limit,
            offset=filters.offset,
            start_date=filters.start_date,
            end_date=filters.end_date,
            annotation_status=filters.annotation_status,
            label_ids=label_ids_str,
            subsets=filters.subsets,
            exclude_types=exclude_types,
            sort_by=filters.sort_by,
            sort_direction=filters.sort_direction,
        )
        return [
            Video.model_validate(media_db).model_copy(
                update={"annotated_frame_count": repo.count_annotated_video_frames_by_video_id(media_db.id)}
            )
            if media_db.type == MediaType.VIDEO
            else MediaAdapter.validate_python(media_db)
            for media_db in media_dbs
        ]

    def get_media_by_id(self, project_id: UUID, media_id: UUID) -> Media:
        """Get a media by its ID"""
        repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        db_media = repo.get_by_id(str(media_id))
        if not db_media:
            raise ResourceNotFoundError(ResourceType.MEDIA, str(media_id))
        return (
            Video.model_validate(db_media).model_copy(
                update={"annotated_frame_count": repo.count_annotated_video_frames_by_video_id(db_media.id)}
            )
            if db_media.type == MediaType.VIDEO
            else MediaAdapter.validate_python(db_media)
        )

    def get_media_by_ids(self, project_id: UUID, media_ids: list[UUID]) -> list[Media]:
        """Get a media list by its IDs"""
        repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        db_media_list = repo.get_by_ids([str(media_id) for media_id in media_ids])
        result = []
        for media_id in media_ids:
            db_media = next((m for m in db_media_list if m.id == str(media_id)), None)
            if not db_media:
                raise ResourceNotFoundError(ResourceType.MEDIA, str(media_id))
            result.append(MediaAdapter.validate_python(db_media))
        return result

    def get_media_binary_path(self, project_id: UUID, media: MediaDB | Media) -> Path:
        dataset_dir = self.projects_dir / f"{project_id}/dataset"
        return dataset_dir / f"{media.id}.{media.format}"

    def get_media_binary_path_by_id(self, project_id: UUID, media_id: UUID) -> Path:
        """Get a media binary content by its ID"""
        media = self.get_media_by_id(project_id=project_id, media_id=media_id)
        return self.get_media_binary_path(project_id=project_id, media=media)

    def get_media_thumbnail_path(self, project: Project, media: MediaDB | Media) -> Path:
        """Get a media thumbnail binary content"""
        return self.get_media_thumbnail_path_by_id(project_id=project.id, media_id=media.id)

    def get_media_thumbnail_path_by_id(self, project_id: UUID, media_id: UUID | str) -> Path:
        """Get a media thumbnail binary content"""
        return self.projects_dir / f"{project_id}/dataset/{media_id}-thumb.jpg"

    @staticmethod
    def _crop_image_to_thumbnail(image: Image.Image) -> Image.Image:
        """Regenerate an image thumbnail by its path"""
        thumbnail = crop_to_thumbnail(
            image=image, target_width=DEFAULT_THUMBNAIL_SIZE, target_height=DEFAULT_THUMBNAIL_SIZE
        )
        return convert_to_jpeg_compatible(thumbnail)

    def delete_media(self, project: Project, media_id: UUID) -> None:
        """Delete a media by its ID"""
        media = self.get_media_by_id(project_id=project.id, media_id=media_id)
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)

        binary_path = self.get_media_binary_path(project_id=project.id, media=media)
        try:
            os.remove(binary_path)
        except FileNotFoundError:
            logger.warning("Media {} binary was not found during deletion", media_id)
        thumbnail_path = self.get_media_thumbnail_path(project=project, media=media)
        try:
            os.remove(thumbnail_path)
        except FileNotFoundError:
            logger.warning("Media {} thumbnail was not found during deletion", media_id)

        repo.delete(obj_id=str(media.id))

    def get_frame_binary(self, project_id: UUID, video: Video, frame_index: int) -> Image.Image:
        video_path = self.get_media_binary_path(project_id=project_id, media=video)
        return self._get_frame_binary_from_video_file(video_path=video_path, frame_index=frame_index)

    def get_frame_binaries(self, project: Project, video: Video, frame_indexes: list[int]) -> dict[int, np.ndarray]:
        """
        Extract multiple frames from a video in a single pass.

        Args:
            project: Project containing the video.
            video: Video to extract frames from.
            frame_indexes: List of frame indexes to extract.

        Returns:
            Dictionary mapping frame index to numpy array (RGB format).
        """
        video_path = self.get_media_binary_path(project_id=project.id, media=video)
        return self._get_video_service().extract_video_frames(video_path=video_path, frame_indexes=frame_indexes)

    def get_frame_thumbnail(self, project: Project, video: Video, frame_index: int) -> Image.Image:
        video_frame = self.get_frame_binary(project_id=project.id, video=video, frame_index=frame_index)
        return MediaService._crop_image_to_thumbnail(video_frame)

    def _get_frame_binary_from_video_file(self, video_path: Path, frame_index: int) -> Image.Image:
        video_frame_numpy = self._get_video_service().extract_video_frame(
            video_path=video_path, frame_index=frame_index
        )
        return MediaService._read_image_from_ndarray(video_frame_numpy)

    def save_video_frame(
        self, project: Project, video: Video, frame_index: int, frame_image: Image.Image
    ) -> VideoFrame:
        """Saves video frame with specified video ID and frame_index"""
        media_id = uuid4()
        format = ImageFormat.JPG

        dataset_dir = self.projects_dir / f"{project.id}/dataset"
        video_frame_path = dataset_dir / f"{media_id}.{format}"

        frame_image.save(video_frame_path)

        try:
            MediaService._generate_and_save_thumbnail(image=frame_image, path=dataset_dir / f"{media_id}-thumb.jpg")

            db_video_frame = MediaDB(
                id=str(media_id),
                project_id=str(project.id),
                type=MediaType.VIDEO_FRAME,
                name=f"{video.name}_frame_{frame_index}",
                format=str(format),
                width=frame_image.width,
                height=frame_image.height,
                video_id=str(video.id),
                frame_index=frame_index,
                size=os.path.getsize(video_frame_path),
                source_id=None,
            )

            repo = MediaRepository(project_id=str(project.id), db=self.db_session)
            db_video_frame = repo.save(db_video_frame)
        except Exception as e:
            video_frame_path.unlink(missing_ok=True)
            raise e
        return VideoFrame.model_validate(db_video_frame, from_attributes=True)

    def get_video_frame_by_video_id_and_index(
        self, project: Project, video_id: UUID, frame_index: int
    ) -> VideoFrame | None:
        """
        Returns annotated video frame by video ID and frame index.

        Args:
            project: Project
            video_id: Video identifier
            frame_index: Frame index

        Returns:
            Video frame data if such frame has been annotated, None otherwise.
        """
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        db_media = repo.get_video_frame_by_video_id_and_index(video_id=str(video_id), frame_index=frame_index)
        return VideoFrame.model_validate(db_media, from_attributes=True) if db_media else None

    def search_video_frames_by_video_id_and_indexes(
        self, project: Project, video_id: UUID, frame_indexes: list[int]
    ) -> list[VideoFrame]:
        """
        Search annotated video frames by video ID and frame indexes list.
        If certain frame index is not found, then it will be missing in the resulting list.

        Args:
            project: Project
            video_id: Video identifier
            frame_indexes: Frame indexes list

        Returns:
            Video frames data
        """
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        db_media_list = repo.search_video_frames_by_video_id_and_indexes(
            video_id=str(video_id), frame_indexes=frame_indexes
        )
        return [VideoFrame.model_validate(db_media, from_attributes=True) for db_media in db_media_list]

    def list_annotated_video_frames_by_video_id(
        self, project: Project, video_id: UUID, frame_index_from: int = 0, frame_index_to: int = 10
    ) -> list[tuple[DatasetItem, VideoFrame]]:
        """
        Returns all annotated video frames falling into the specified frame index range.

        Args:
            project: Project
            video_id: Video identifier
            frame_index_from: Frame index range start, default is 0
            frame_index_to: Frame index range end, default is 10

        Returns:
            Annotated video frames list with corresponding dataset items.
        """
        repo = MediaRepository(project_id=str(project.id), db=self.db_session)
        db_media_list = repo.list_annotated_video_frames_by_video_id(
            video_id=str(video_id), frame_index_from=frame_index_from, frame_index_to=frame_index_to
        )
        return [
            (DatasetItem.model_validate(db_dataset_item), VideoFrame.model_validate(db_media, from_attributes=True))
            for (db_dataset_item, db_media) in db_media_list
        ]
