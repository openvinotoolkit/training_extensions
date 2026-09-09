# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Generator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.api.validators import DatasetRevisionID, DatasetViewID, ProjectID, SinkID, SourceID
from app.core.jobs.control_plane import JobQueue
from app.db import get_db_session
from app.models import Project, Sink, Source
from app.models.dataset_revision import DatasetRevision
from app.models.dataset_view import DatasetView
from app.scheduler import Scheduler
from app.services import (
    BaseWeightsService,
    DatasetRevisionService,
    DatasetService,
    DatasetViewService,
    LabelService,
    MediaNumpyLoader,
    MediaPredictionService,
    MediaSegmentService,
    MediaService,
    MetricsService,
    ModelService,
    PipelineMetricsService,
    PipelineService,
    ProjectService,
    SinkService,
    SourceMediaService,
    SourceUpdateService,
    StagedDatasetService,
    SystemService,
)
from app.services.data_collect import DataCollector
from app.services.demo_files_service import DemoFilesService
from app.services.event.event_bus import EventBus
from app.services.inference import InferenceServer
from app.services.inference_status_service import InferenceStatusService
from app.services.license_service import LicenseService
from app.services.sink_status_service import SinkStatusService
from app.services.source_status_service import SourceStatusService
from app.services.training_configuration_service import TrainingConfigurationService
from app.services.video import VideoService
from app.webrtc.manager import WebRTCManager


def get_file_name_and_extension(file: UploadFile) -> tuple[str, str]:
    """Return the file name and extension"""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="File name cannot be empty.")
    full_name = file.filename.strip()
    if not full_name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="File name cannot be empty.")
    file_name, file_ext = os.path.splitext(full_name)
    file_name = file_name.strip()  # remove whitespace characters between the basename and the extension
    file_ext = file_ext[1:]  # remove leading dot in the extension
    if not file_ext:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="File extension cannot be empty.")
    return file_name, file_ext


def get_file_size(file: UploadFile) -> int:
    """Return the file size in bytes"""
    if not file.size:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="File size should be defined.")
    return file.size


def get_db() -> Generator[Session]:
    """Provides a database session."""
    with get_db_session() as session:
        yield session


def get_scheduler(request: Request) -> Scheduler:
    """Provides the global Scheduler instance."""
    return request.app.state.scheduler


def get_data_dir(request: Request) -> Path:
    """Provides the path to the folder that stores the projects data. This path is defined in the app settings."""
    return request.app.state.settings.data_dir


def get_job_dir(request: Request) -> Path:
    """Provides the path to the folder where the jobs logs are saved. This path is defined in the app settings."""
    return request.app.state.settings.job_dir


def get_staged_datasets_dir(request: Request) -> Path:
    """Provides the path to the folder where the staged datasets are saved. This path is defined in the app settings."""
    return request.app.state.settings.staged_datasets_dir


def get_source_media_dir(request: Request) -> Path:
    """Provides the path to the folder where uploaded source videos are saved. Defined in the app settings."""
    return request.app.state.settings.source_media_dir


def get_ice_servers(request: Request) -> list[dict]:
    """Provides the ICE servers from settings."""
    return request.app.state.settings.ice_servers


def get_inference_media_limit(request: Request) -> int:
    """Provides the inference media limit from settings."""
    return request.app.state.settings.inference_media_limit


def get_inference_model_ttl(request: Request) -> int:
    """Provides the inference model TTL from settings."""
    return request.app.state.settings.inference_model_ttl


def get_inference_keyframe_stride(request: Request) -> int:
    """Provides the inference frame skip from settings."""
    return request.app.state.settings.inference_keyframe_stride


def get_sam_encoder_xml_path(request: Request) -> Path:
    """Provides the path to the SAM encoder model XML file. This path is defined in the app settings."""
    return request.app.state.settings.sam_encoder_xml_path


def get_sam_ov_cache_path(request: Request) -> Path | None:
    """Provides the OpenVINO cache directory for the SAM encoder. This path is defined in the app settings."""
    return request.app.state.settings.sam_ov_cache_path


def get_event_bus(request: Request) -> EventBus:
    """Provides an EventBus instance."""
    return request.app.state.event_bus


def get_data_collector(request: Request) -> DataCollector:
    """Provides an DataCollector instance."""
    return request.app.state.data_collector


def get_inference_server(request: Request) -> InferenceServer:
    """Provides an InferenceServer instance."""
    return request.app.state.inference_server


def get_video_service(request: Request) -> VideoService:
    """Provides the VideoService instance from application state."""
    return request.app.state.video_service


def get_metrics_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> MetricsService:
    """Provides a MetricsService instance for collecting and retrieving metrics."""
    return MetricsService(scheduler.shm_metrics.name, scheduler.shm_metrics_lock)


def get_sink_service(
    event_bus: Annotated[EventBus, Depends(get_event_bus)], db: Annotated[Session, Depends(get_db)]
) -> SinkService:
    """Provides a SinkService instance."""
    return SinkService(event_bus=event_bus, db_session=db)


def get_source_media_service(
    source_media_dir: Annotated[Path, Depends(get_source_media_dir)],
) -> SourceMediaService:
    """Provides a SourceMediaService instance for storing uploaded source videos."""
    return SourceMediaService(source_media_dir)


def get_source_update_service(
    event_bus: Annotated[EventBus, Depends(get_event_bus)],
    db: Annotated[Session, Depends(get_db)],
    source_media_service: Annotated[SourceMediaService, Depends(get_source_media_service)],
) -> SourceUpdateService:
    """Provides a SourceUpdateService instance."""
    return SourceUpdateService(event_bus=event_bus, db_session=db, source_media_service=source_media_service)


def get_source_status_service(
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> SourceStatusService:
    """Provides a SourceStatusService instance."""
    return SourceStatusService(
        source_status_shm=scheduler.source_status_shm, source_status_lock=scheduler.source_status_lock
    )


def get_sink_status_service(
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> SinkStatusService:
    """Provides a SinkStatusService instance."""
    return SinkStatusService(sink_status_holder=scheduler.sink_status_holder)


def get_inference_status_service(
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> InferenceStatusService:
    """Provides an InferenceStatusService instance."""
    return InferenceStatusService(
        inference_status_shm=scheduler.inference_status_shm, inference_status_lock=scheduler.inference_status_lock
    )


def get_system_service() -> SystemService:
    """Provides a SystemService instance for system-level operations."""
    return SystemService()


def get_model_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    db: Annotated[Session, Depends(get_db)],
) -> ModelService:
    """Provides a ModelService instance with the model reload event from the scheduler."""
    return ModelService(data_dir=data_dir, db_session=db)


def get_pipeline_service(
    event_bus: Annotated[EventBus, Depends(get_event_bus)],
    db: Annotated[Session, Depends(get_db)],
    system_service: Annotated[SystemService, Depends(get_system_service)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> PipelineService:
    """Provides a PipelineService instance ."""
    return PipelineService(
        event_bus=event_bus, db_session=db, system_service=system_service, model_service=model_service
    )


def get_pipeline_metrics_service(
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> PipelineMetricsService:
    """Provides a PipelineMetricsService instance."""
    return PipelineMetricsService(
        pipeline_service=pipeline_service,
        metrics_service=metrics_service,
    )


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager


def get_label_service(db: Annotated[Session, Depends(get_db)]) -> LabelService:
    """Provides a LabelService instance for managing labels."""
    return LabelService(db_session=db)


def get_project_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    db: Annotated[Session, Depends(get_db)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    label_service: Annotated[LabelService, Depends(get_label_service)],
) -> ProjectService:
    """Provides a ProjectService instance for managing projects."""
    return ProjectService(
        data_dir=data_dir, label_service=label_service, pipeline_service=pipeline_service, db_session=db
    )


def get_media_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    video_service: Annotated[VideoService, Depends(get_video_service)],
    db: Annotated[Session, Depends(get_db)],
) -> MediaService:
    """Provides a MediaService instance."""
    return MediaService(data_dir=data_dir, video_service=video_service, db_session=db)


def get_media_numpy_loader(
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> MediaNumpyLoader:
    """Provides a MediaNumpyLoader instance."""
    return MediaNumpyLoader(media_service=media_service)


def get_dataset_service(
    label_service: Annotated[LabelService, Depends(get_label_service)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    db: Annotated[Session, Depends(get_db)],
) -> DatasetService:
    """Provides a DatasetService instance."""
    return DatasetService(label_service=label_service, media_service=media_service, db_session=db)


def get_media_prediction_service(
    label_service: Annotated[LabelService, Depends(get_label_service)],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    inference_server: Annotated[InferenceServer, Depends(get_inference_server)],
    inference_model_ttl: Annotated[int, Depends(get_inference_model_ttl)],
    inference_keyframe_stride: Annotated[int, Depends(get_inference_keyframe_stride)],
    media_numpy_loader: Annotated[MediaNumpyLoader, Depends(get_media_numpy_loader)],
    db: Annotated[Session, Depends(get_db)],
) -> MediaPredictionService:
    """Provides a MediaPredictionService instance."""
    return MediaPredictionService(
        label_service=label_service,
        media_service=media_service,
        inference_server=inference_server,
        inference_model_ttl=inference_model_ttl,
        inference_keyframe_stride=inference_keyframe_stride,
        media_numpy_loader=media_numpy_loader,
        db_session=db,
    )


def get_media_segment_service(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    media_numpy_loader: Annotated[MediaNumpyLoader, Depends(get_media_numpy_loader)],
    model_xml_path: Annotated[Path, Depends(get_sam_encoder_xml_path)],
    ov_cache_path: Annotated[Path | None, Depends(get_sam_ov_cache_path)],
    db: Annotated[Session, Depends(get_db)],
) -> MediaSegmentService:
    """Provides a MediaSegmentService instance."""
    return MediaSegmentService(
        media_service=media_service,
        media_numpy_loader=media_numpy_loader,
        model_xml_path=model_xml_path,
        ov_cache_path=ov_cache_path,
        db_session=db,
    )


def get_dataset_revision_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    db: Annotated[Session, Depends(get_db)],
) -> DatasetRevisionService:
    """Provides a DatasetRevisionService instance."""
    return DatasetRevisionService(data_dir=data_dir, db_session=db)


def get_demo_files_service(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    label_service: Annotated[LabelService, Depends(get_label_service)],
) -> DemoFilesService:
    """Provides a DemoFilesService instance."""
    return DemoFilesService(media_service=media_service, label_service=label_service)


def get_project(
    project_id: ProjectID,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Provides a ProjectView instance for request scoped project."""
    return project_service.get_project_by_id(project_id)


def get_sink(
    sink_id: SinkID,
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> Sink:
    """Provides a Sink instance for request scoped sink."""
    return sink_service.get_by_id(sink_id)


def get_source(
    source_id: SourceID,
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> Source:
    """Provides a Source instance for request scoped source."""
    return source_update_service.get_by_id(source_id)


def get_base_weights_service(data_dir: Annotated[Path, Depends(get_data_dir)]) -> BaseWeightsService:
    """Provides a BaseWeightsService instance for managing base weights."""
    return BaseWeightsService(data_dir)


def get_license_service(
    data_dir: Annotated[Path, Depends(get_data_dir)],
    request: Request,
) -> LicenseService:
    """Provides a LicenseService instance for tracking license consent."""
    return LicenseService(data_dir=data_dir, app_version=request.app.state.settings.version)


def get_staged_dataset_service(
    staged_datasets_dir: Annotated[Path, Depends(get_staged_datasets_dir)],
) -> StagedDatasetService:
    """Provides a StagedDatasetService instance for managing staged datasets."""
    return StagedDatasetService(staged_datasets_dir)


def get_job_queue(request: Request) -> JobQueue:
    """
    Provides the global JobQueue instance from FastAPI application's state.
    The JobQueue is responsible for managing job submissions and tracking job statuses.
    """
    return request.app.state.job_queue


def get_training_configuration_service(db: Annotated[Session, Depends(get_db)]) -> TrainingConfigurationService:
    """Provides a TrainingConfigurationService instance for managing training configurations."""
    return TrainingConfigurationService(db_session=db)


def get_dataset_revision(
    project_id: ProjectID,
    dataset_revision_id: DatasetRevisionID,
    dataset_revision_service: Annotated[DatasetRevisionService, Depends(get_dataset_revision_service)],
) -> DatasetRevision:
    """Provides a DatasetService instance."""
    return dataset_revision_service.get_dataset_revision(project_id=project_id, revision_id=dataset_revision_id)


def get_dataset_view_service(db: Annotated[Session, Depends(get_db)]) -> DatasetViewService:
    """Provides a DatasetViewService instance."""
    return DatasetViewService(db_session=db)


def get_dataset_view(
    project_id: ProjectID,
    dataset_view_id: DatasetViewID,
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> DatasetView:
    """Provides a DatasetView instance for a request scoped dataset view."""
    return dataset_view_service.get_dataset_view_by_id(project_id=project_id, dataset_view_id=dataset_view_id)
