# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.openapi.models import Example
from starlette.responses import Response

from app.api.dependencies import (
    get_dataset_service,
    get_dataset_view_service,
    get_file_name_and_extension,
    get_inference_media_limit,
    get_media_prediction_service,
    get_media_segment_service,
    get_media_service,
    get_project,
    get_project_service,
    get_system_service,
)
from app.api.io_utils import write_bytes_to_response, write_file_to_response, write_image_to_response
from app.api.schemas.media import (
    AnnotatedVideoFrame,
    BulkDeleteMedia,
    MediaAnnotations,
    MediaView,
    MediaViewAdapter,
    MediaWithPagination,
    SetMediaAnnotations,
)
from app.api.validators import MediaID, ProjectID, normalize_datetime_to_utc
from app.core.models import Pagination
from app.models import BatchInferenceResult, DatasetItemAnnotationStatus, DatasetItemSubset, Media, Project, Video
from app.models.media import (
    ImageFormat,
    MediaListPredictionRequest,
    MediaSortBy,
    MediaType,
    NotAnnotatedVideoFrame,
    SortDirection,
    VideoFormat,
)
from app.services import (
    DatasetService,
    DatasetViewService,
    MediaPredictionService,
    MediaService,
    ProjectService,
    SystemService,
)
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.dataset_service import AnnotationValidationError, SubsetAlreadyAssignedError
from app.services.inference import InferenceBusyError
from app.services.media_numpy_loader import BinaryNotFoundError
from app.services.media_prediction_service import VideoRangeError
from app.services.media_service import ImageMetadata, InvalidImageError, MediaFilters
from app.services.sam import MediaSegmentService
from app.utils.images import needs_display_normalization, normalize_image_to_png_bytes

router = APIRouter(prefix="/api/projects/{project_id}/dataset/media", tags=["Media"])

DEFAULT_MEDIA_NUMBER_RETURNED = 10
MAX_MEDIA_NUMBER_RETURNED = 100

DEFAULT_FRAME_INDEX_FROM = 0
DEFAULT_FRAME_INDEX_TO = 50

SET_MEDIA_ANNOTATIONS_BODY_EXAMPLES = {
    "single_label": Example(
        summary="Single label (multiclass classification)",
        value={
            "annotations": [
                {"labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}], "shape": {"type": "full_image"}}
            ],
        },
    ),
    "multi_label": Example(
        summary="Multiple labels (multilabel classification)",
        value={
            "annotations": [
                {
                    "labels": [
                        {"id": "d476573e-d43c-42a6-9327-199a9aa75c33"},
                        {"id": "bbb782b7-8322-44e8-b6a9-90a5c9ee4bad"},
                    ],
                    "shape": {"type": "full_image"},
                },
            ],
        },
    ),
    "bounding_boxes": Example(
        summary="Bounding boxes (detection)",
        value={
            "annotations": [
                {
                    "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                    "shape": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200},
                },
                {
                    "labels": [{"id": "bbb782b7-8322-44e8-b6a9-90a5c9ee4bad"}],
                    "shape": {"type": "rectangle", "x": 150, "y": 250, "width": 80, "height": 120},
                },
            ],
        },
    ),
    "polygons": Example(
        summary="Polygons (segmentation)",
        value={
            "annotations": [
                {
                    "labels": [{"id": "d476573e-d43c-42a6-9327-199a9aa75c33"}],
                    "shape": {"type": "polygon", "points": [[10, 20], [20, 60], [30, 40]]},
                },
                {
                    "labels": [{"id": "bbb782b7-8322-44e8-b6a9-90a5c9ee4bad"}],
                    "shape": {"type": "polygon", "points": [[150, 250], [180, 300], [200, 280]]},
                },
            ],
        },
    ),
    "empty": Example(
        summary="No objects (detection / segmentation)",
        value={
            "annotations": [],
        },
    ),
}

BULK_MEDIA_DELETE_EXAMPLE = {
    "list_of_media_ids": Example(
        summary="List of Media IDs",
        value={"media_ids": ["d476573e-d43c-42a6-9327-199a9aa75c33", "bbb782b7-8322-44e8-b6a9-90a5c9ee4bad"]},
    )
}


def _parse_media_format(extension: str) -> ImageFormat | VideoFormat:
    try:
        return ImageFormat[extension.upper()]
    except KeyError:
        try:
            return VideoFormat[extension.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unsupported media extension: {extension}",
            )


def _get_request_media(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> Media | NotAnnotatedVideoFrame:
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    if media.type != MediaType.VIDEO or frame_index is None:
        return media

    # Video frames can be identified by video ID and frame index
    if frame_index >= media.frame_count:  # pyrefly: ignore[unsupported-operation]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video frame index {frame_index} exceeds video frames count {media.frame_count}.",
        )
    video_frame = media_service.get_video_frame_by_video_id_and_index(
        project=project, video_id=media_id, frame_index=frame_index
    )
    return video_frame if video_frame is not None else NotAnnotatedVideoFrame(video=media, frame_index=frame_index)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=MediaView,
    responses={
        status.HTTP_201_CREATED: {"description": "Media created"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid media has been uploaded"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def add_media(
    project: Annotated[Project, Depends(get_project)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    file_name_and_extension: Annotated[tuple[str, str], Depends(get_file_name_and_extension)],
    file: Annotated[UploadFile, File()],
) -> MediaView:
    """Add a new media to the dataset by uploading an image or a video"""
    name, extension = file_name_and_extension
    media_format = _parse_media_format(extension)
    try:
        if isinstance(media_format, ImageFormat):
            media = media_service.create_image(
                ImageMetadata(
                    project_id=project.id,
                    data=file.file,
                    name=name,
                    image_format=media_format,
                )
            )
            dataset_service.create_dataset_item(
                project_id=project.id,
                task=project.task,
                media=media,
                user_reviewed=False,
            )
        else:
            # Dataset items for videos are created separately after video upload for each frame being annotated
            media = media_service.create_video(
                project_id=project.id,
                data=file.file,
                name=name,
                video_format=media_format,
            )
        return MediaViewAdapter.validate_python(media, from_attributes=True)
    except InvalidImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid image has been uploaded."
        )


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available media", "model": MediaWithPagination},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def list_media(  # noqa: PLR0913
    project: Annotated[Project, Depends(get_project)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_MEDIA_NUMBER_RETURNED)] = DEFAULT_MEDIA_NUMBER_RETURNED,
    offset: Annotated[int, Query(ge=0, le=2_147_483_647)] = 0,
    start_date: Annotated[datetime | None, Query()] = None,
    end_date: Annotated[datetime | None, Query()] = None,
    annotation_status: Annotated[DatasetItemAnnotationStatus | None, Query()] = None,
    labels: Annotated[list[UUID] | None, Query()] = None,
    subsets: Annotated[list[DatasetItemSubset] | None, Query()] = None,
    sort_by: Annotated[MediaSortBy, Query()] = MediaSortBy.UPLOAD_DATE,
    sort_direction: Annotated[SortDirection, Query()] = SortDirection.DESC,
    dataset_view_id: Annotated[
        UUID | None,
        Query(description="If provided, list media assigned to this dataset view instead of the entire dataset"),
    ] = None,
) -> MediaWithPagination:
    """
    List the available media and their metadata. This endpoint supports pagination.

    This is also the endpoint used to list the content of a dataset view: when `dataset_view_id` is provided,
    only the media assigned to that view are returned, with the same filtering, sorting and pagination options.
    """
    start_date = normalize_datetime_to_utc(start_date)
    end_date = normalize_datetime_to_utc(end_date)

    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Start date must be before end date."
        )
    subset_values = [item.value for item in subsets] if subsets else None
    filters = MediaFilters(
        limit=limit,
        offset=offset,
        start_date=start_date,
        end_date=end_date,
        annotation_status=annotation_status,
        label_ids=labels,
        subsets=subset_values,
        sort_by=sort_by,
        sort_direction=sort_direction,
    )
    if dataset_view_id is not None:
        total = dataset_view_service.count_dataset_view_media(
            project_id=project.id, dataset_view_id=dataset_view_id, filters=filters
        )
        media_list = dataset_view_service.list_dataset_view_media(
            project_id=project.id, dataset_view_id=dataset_view_id, filters=filters
        )
    else:
        total = media_service.count_media(
            project=project,
            start_date=start_date,
            end_date=end_date,
            annotation_status=annotation_status,
            label_ids=labels,
            subsets=subset_values,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        media_list = media_service.list_media(
            project_id=project.id,
            filters=filters,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
    return MediaWithPagination(
        items=[MediaViewAdapter.validate_python(media, from_attributes=True) for media in media_list],
        pagination=Pagination(
            limit=limit,
            offset=offset,
            total=total,
            count=len(media_list),
        ),
    )


@router.get(
    "/{media_id}",
    responses={
        status.HTTP_200_OK: {"description": "Media item found", "model": MediaView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media or project not found"},
    },
)
def get_media(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> MediaView:
    """Get information about a specific media"""
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    return MediaViewAdapter.validate_python(media, from_attributes=True)


@router.get(
    "/{media_id}/frames",
    responses={
        status.HTTP_200_OK: {"description": "List of annotated video frames", "model": list[AnnotatedVideoFrame]},
        status.HTTP_404_NOT_FOUND: {"description": "Project or video not found"},
    },
)
def list_video_frames(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
    frame_index_from: Annotated[int, Query(ge=0)] = DEFAULT_FRAME_INDEX_FROM,
    frame_index_to: Annotated[int, Query(ge=0)] = DEFAULT_FRAME_INDEX_TO,
) -> list[AnnotatedVideoFrame]:
    """List annotated video frames with frame index range"""
    media = media_service.get_media_by_id(project_id=project.id, media_id=media_id)
    if media.type != MediaType.VIDEO:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Requested media is not video.")
    annotated_video_frames = media_service.list_annotated_video_frames_by_video_id(
        project=project,
        video_id=media_id,
        frame_index_from=frame_index_from,
        frame_index_to=frame_index_to,
    )
    return [
        AnnotatedVideoFrame(
            media_id=video_frame.id,
            frame_index=video_frame.frame_index,
            annotation_data=MediaAnnotations(
                annotations=dataset_item.annotation_data,  # type: ignore[arg-type]
                prediction_model_id=dataset_item.prediction_model_id,
                user_reviewed=dataset_item.user_reviewed,
                subset=dataset_item.subset,
            ),
        )
        for (dataset_item, video_frame) in annotated_video_frames
    ]


@router.get(
    "/{media_id}/binary",
    response_model=None,
    responses={
        status.HTTP_200_OK: {"description": "Media binary found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, media binary or project not found"},
    },
)
def get_media_binary(
    project: Annotated[Project, Depends(get_project)],
    media: Annotated[Media | NotAnnotatedVideoFrame, Depends(_get_request_media)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    raw: Annotated[bool, Query(description="Return the original file without normalization")] = False,
) -> Response:
    """Get media binary content"""
    if isinstance(media, NotAnnotatedVideoFrame):
        frame_binary = media_service.get_frame_binary(
            project_id=project.id, video=media.video, frame_index=media.frame_index
        )
        return write_image_to_response(
            image=frame_binary, filename=f"{media.video.name}_frame_{media.frame_index}.jpeg"
        )

    binary_path = media_service.get_media_binary_path_by_id(project_id=project.id, media_id=media.id)

    # Normalize high bit depth images on the fly for display; cheap mode-check first
    if (
        not raw
        and media.type == MediaType.IMAGE
        and isinstance(media.format, ImageFormat)
        and needs_display_normalization(binary_path)
    ):
        png_bytes = normalize_image_to_png_bytes(binary_path)
        return write_bytes_to_response(content=png_bytes, filename=f"{media.name}.png", media_type="image/png")

    filename = f"{media.name}.{media.format.value.lower()}"

    return write_file_to_response(path=binary_path, filename=filename)


@router.get(
    "/{media_id}/thumbnail",
    response_model=None,
    responses={
        status.HTTP_200_OK: {"description": "Media thumbnail found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, media thumbnail or project not found"},
    },
)
def get_media_thumbnail(
    project_id: ProjectID,
    media_id: MediaID,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> Response:
    """Get media thumbnail binary content"""

    def _response(thumbnail_path: Path, media_id: str | UUID) -> Response:
        return write_file_to_response(
            path=thumbnail_path, filename=f"{str(media_id)}-thumb.jpeg", cache_control="public, max-age=31536000"
        )

    def _not_found() -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media thumbnail file is not found.")

    thumbnail_path = media_service.get_media_thumbnail_path_by_id(project_id=project_id, media_id=media_id)
    if frame_index is None and os.path.exists(thumbnail_path):
        return _response(thumbnail_path=thumbnail_path, media_id=media_id)

    project = project_service.get_project_by_id(project_id=project_id)
    media = media_service.get_media_by_id(project_id=project_id, media_id=media_id)
    if media.type != MediaType.VIDEO or frame_index is None:
        raise _not_found()

    # Video frames can be identified by video ID and frame index
    if frame_index >= media.frame_count:  # pyrefly: ignore[unsupported-operation]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video frame index {frame_index} exceeds video frames count {media.frame_count}.",
        )

    video_frame = media_service.get_video_frame_by_video_id_and_index(
        project=project, video_id=media_id, frame_index=frame_index
    )
    if video_frame is not None:
        thumbnail_path = media_service.get_media_thumbnail_path_by_id(project_id=project_id, media_id=video_frame.id)
        if os.path.exists(thumbnail_path):
            return _response(thumbnail_path=thumbnail_path, media_id=video_frame.id)
        raise _not_found()

    # If the media is a video frame, generate the thumbnail on the fly
    frame_thumbnail = media_service.get_frame_thumbnail(project=project, video=media, frame_index=frame_index)
    return write_image_to_response(image=frame_thumbnail, filename=f"{media.name}_frame_{frame_index}.jpeg")


@router.delete(
    "/{media_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Media deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media or project not found"},
    },
)
def delete_media(
    project: Annotated[Project, Depends(get_project)],
    media_id: MediaID,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> None:
    """Delete media from the dataset"""
    media_service.delete_media(project=project, media_id=media_id)


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Requested media has been deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or invalid annotation content"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def bulk_delete_media(
    project: Annotated[Project, Depends(get_project)],
    media_ids_list: Annotated[BulkDeleteMedia, Body(openapi_examples=BULK_MEDIA_DELETE_EXAMPLE)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> None:
    """Bulk delete media"""
    for media_id in media_ids_list.media_ids:
        try:
            media_service.delete_media(project=project, media_id=media_id)
        except ResourceNotFoundError as error:
            # Ignore not-found errors only for media resources; re-raise for others (e.g., project)
            if getattr(error, "resource_type", None) != ResourceType.MEDIA:
                raise


@router.post(
    "/{media_id}/annotations",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Annotation created or updated", "model": MediaAnnotations},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or invalid annotation content"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, dataset item or project not found"},
        status.HTTP_409_CONFLICT: {"description": "Dataset item already has a subset assigned"},
    },
)
def set_media_annotations(
    project: Annotated[Project, Depends(get_project)],
    media: Annotated[Media | NotAnnotatedVideoFrame, Depends(_get_request_media)],
    media_annotations: Annotated[SetMediaAnnotations, Body(openapi_examples=SET_MEDIA_ANNOTATIONS_BODY_EXAMPLES)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> MediaAnnotations:
    """Set media annotations"""
    if isinstance(media, Video) and not frame_index:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video frame index is not provided.")
    if isinstance(media, NotAnnotatedVideoFrame):
        frame_binary = media_service.get_frame_binary(
            project_id=project.id, video=media.video, frame_index=media.frame_index
        )
        video_frame = media_service.save_video_frame(
            project=project, video=media.video, frame_index=media.frame_index, frame_image=frame_binary
        )
        dataset_service.create_dataset_item(
            project_id=project.id,
            task=project.task,
            media=video_frame,
            user_reviewed=True,
        )
        media = video_frame
    # Dataset item has the same ID as media
    dataset_item_id = media.id
    try:
        dataset_item = dataset_service.set_dataset_item_annotations(
            project=project,
            dataset_item_id=dataset_item_id,
            annotations=media_annotations.annotations,
            # Annotations submitted via API are considered user-reviewed, unlike auto-generated predictions
            user_reviewed=True,
            prediction_model_id=None,
        )
    except AnnotationValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if media_annotations.subset is not None:
        try:
            dataset_item = dataset_service.assign_dataset_item_subset(
                project_id=project.id, dataset_item_id=dataset_item_id, subset=media_annotations.subset
            )
        except SubsetAlreadyAssignedError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    return MediaAnnotations(
        media_id=media.id,
        annotations=dataset_item.annotation_data,  # type: ignore[arg-type]
        prediction_model_id=dataset_item.prediction_model_id,
        user_reviewed=dataset_item.user_reviewed,
        subset=dataset_item.subset,
    )


@router.get(
    "/{media_id}/annotations",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Annotation found", "model": MediaAnnotations},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {
            "description": "Media, dataset item or project not found or media is not annotated"
        },
    },
)
def get_media_annotations(
    project: Annotated[Project, Depends(get_project)],
    media: Annotated[Media | NotAnnotatedVideoFrame, Depends(_get_request_media)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> MediaAnnotations:
    """Get the media annotations"""
    if isinstance(media, Video) and not frame_index:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video frame index is not provided.")
    if isinstance(media, NotAnnotatedVideoFrame):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video frame not found for the given index.")
    # Dataset item has the same ID as media
    dataset_item_id = media.id
    dataset_item = dataset_service.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)
    if dataset_item.annotation_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media has not been annotated yet.")
    return MediaAnnotations(
        annotations=dataset_item.annotation_data,
        prediction_model_id=dataset_item.prediction_model_id,
        user_reviewed=dataset_item.user_reviewed,
        subset=dataset_item.subset,
    )


@router.delete(
    "/{media_id}/annotations",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Media annotations deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid media ID or project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, dataset item or project not found"},
    },
)
def delete_media_annotation(
    project: Annotated[Project, Depends(get_project)],
    media: Annotated[Media | NotAnnotatedVideoFrame, Depends(_get_request_media)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
) -> None:
    """Delete media annotations"""
    if isinstance(media, Video) and not frame_index:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video frame index is not provided.")
    if isinstance(media, NotAnnotatedVideoFrame):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video frame not found for the given index.")
    # Dataset item has the same ID as media
    dataset_item_id = media.id
    dataset_service.delete_dataset_item_annotations(project=project, dataset_item_id=dataset_item_id)


MEDIA_PREDICT_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_200_OK: {"description": "Media predictions are calculated"},
    status.HTTP_400_BAD_REQUEST: {
        "description": "Missing frame range, range is specified for non-video media, or media inference limit exceeded"
    },
    status.HTTP_404_NOT_FOUND: {"description": "Media, dataset item or project not found"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "description": "Inference server is busy with another request, try again later"
    },
}


@router.post(
    ":predict",
    status_code=status.HTTP_200_OK,
    responses=MEDIA_PREDICT_RESPONSES,
)
@router.post(
    "/media:predict",
    status_code=status.HTTP_200_OK,
    deprecated=True,
    responses=MEDIA_PREDICT_RESPONSES,
)
def media_predict(
    inference_media_limit: Annotated[int, Depends(get_inference_media_limit)],
    project: Annotated[Project, Depends(get_project)],
    request: Annotated[MediaListPredictionRequest, Body()],
    media_prediction_service: Annotated[MediaPredictionService, Depends(get_media_prediction_service)],
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> BatchInferenceResult:
    """Get predictions for media.

    .. deprecated:: The `/media:predict` path is deprecated due to a duplicated `media` path segment
       and will be removed in version 3.4. Use `:predict` instead.
    """
    items_count = sum(
        [
            1
            if media_request.range is None
            else len(
                range(media_request.range.start_frame, media_request.range.end_frame + 1, media_request.range.stride)
            )
            for media_request in request.media
        ]
    )
    if items_count > inference_media_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many media items to predict, requested number is {items_count} "
            f"while limit is {inference_media_limit}. "
            f"Please reduce the number of media or frame range size or set INFERENCE_MEDIA_LIMIT "
            f"environment variable with higher value .",
        )

    try:
        device = system_service.inference_device(request.device)
        return media_prediction_service.predict_media(project=project, request=request, device=device)
    except VideoRangeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except BinaryNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except InferenceBusyError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/{media_id}/embeddings",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Media is segmented, embeddings are calculated",
            "content": {"application/octet-stream": {}},
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Missing frame index or range is specified for non-video media"},
        status.HTTP_404_NOT_FOUND: {"description": "Media, dataset item or project not found"},
        status.HTTP_409_CONFLICT: {"description": "Device is not available on this system for inference"},
    },
)
def get_media_embeddings(
    project: Annotated[Project, Depends(get_project)],
    media: Annotated[Media | NotAnnotatedVideoFrame, Depends(_get_request_media)],
    media_segment_service: Annotated[MediaSegmentService, Depends(get_media_segment_service)],
    system_service: Annotated[SystemService, Depends(get_system_service)],
    frame_index: Annotated[int | None, Query(description="Video frame index", ge=0)] = None,
    device: Annotated[str | None, Query(description="Device to run inference at")] = None,
) -> Response:
    """
    Encode a media through a segment-anything model and return the embeddings as a binary file
    """
    if isinstance(media, Video) and not frame_index:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video frame index is not provided.")
    device = device if device is not None else "cpu"
    if not system_service.is_valid_inference_device(device):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"Device '{device}' is not available on this system"
        )
    try:
        media_id = f"{media.video.id}_{media.frame_index}" if isinstance(media, NotAnnotatedVideoFrame) else media.id
        embeddings_bytes = media_segment_service.encode_media(
            project=project, media=media, device=system_service.inference_device(device)
        )
        return write_bytes_to_response(
            content=embeddings_bytes,
            filename=f"{media_id}_embeddings.safetensors",
            media_type="application/octet-stream",
        )
    except VideoRangeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except BinaryNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
