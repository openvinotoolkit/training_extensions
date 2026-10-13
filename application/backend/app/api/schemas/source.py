# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, TypeAdapter, computed_field

from app.core.models import BaseIDNameModel
from app.models import (
    DisconnectedSourceConfig,
    ImagesFolderSourceConfig,
    IPCameraSourceConfig,
    USBCameraSourceConfig,
    VideoFileSourceConfig,
)
from app.models.source import (
    DisconnectedConfig,
    ImagesFolderConfig,
    IPCameraConfig,
    SourceType,
    USBCameraConfig,
    VideoFileConfig,
)


class DisconnectedSourceConfigView(DisconnectedSourceConfig):
    config_data: DisconnectedConfig = Field(exclude=True)
    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "disconnected",
                "name": "No Source",
                "id": "00000000-0000-0000-0000-000000000000",
            }
        }
    }


class USBCameraSourceConfigView(USBCameraSourceConfig):
    config_data: USBCameraConfig = Field(exclude=True)

    @computed_field
    @property
    def device_id(self) -> int:
        return self.config_data.device_id

    @computed_field
    @property
    def codec(self) -> str | None:
        return self.config_data.codec

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "usb_camera",
                "name": "USB Camera 0",
                "id": "f9e0ae4f-d96c-4304-baab-2ab845362d03",
                "device_id": 0,
                "codec": "YUY2",
            },
            "required": [  # Explicitly exclude "codec" field from required fields
                "name",
                "id",
                "source_type",
                "device_id",
            ],
        }
    }


class IPCameraSourceConfigView(IPCameraSourceConfig):
    config_data: IPCameraConfig = Field(exclude=True)

    @computed_field
    @property
    def stream_url(self) -> str:
        return self.config_data.stream_url

    @computed_field
    @property
    def auth_required(self) -> bool:
        return self.config_data.auth_required

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "ip_camera",
                "name": "Street Camera 123",
                "id": "3d055c8a-2536-46ea-8f3c-832bd6f8bbdc",
                "stream_url": "http://example.com/stream",
                "auth_required": True,
            }
        }
    }


class VideoFileSourceConfigView(VideoFileSourceConfig):
    config_data: VideoFileConfig = Field(exclude=True)

    @computed_field
    @property
    def video_path(self) -> str:
        return self.config_data.video_path

    @computed_field
    @property
    def loop(self) -> bool:
        return self.config_data.loop

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "video_file",
                "name": "Sample Video",
                "id": "712750b2-5a82-47ee-8fba-f3dc96cb615d",
                "video_path": "/path/to/video.mp4",
                "loop": False,
            }
        }
    }


class ImagesFolderSourceConfigView(ImagesFolderSourceConfig):
    config_data: ImagesFolderConfig = Field(exclude=True)

    @computed_field
    @property
    def images_folder_path(self) -> str:
        return self.config_data.images_folder_path

    @computed_field
    @property
    def ignore_existing_images(self) -> bool:
        return self.config_data.ignore_existing_images

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "images_folder",
                "name": "Best Photos",
                "id": "4a580a0e-b841-4c70-bf88-2d68a28f780d",
                "images_folder_path": "/path/to/images",
                "ignore_existing_images": True,
            }
        }
    }


SourceView = Annotated[
    DisconnectedSourceConfigView
    | USBCameraSourceConfigView
    | IPCameraSourceConfigView
    | VideoFileSourceConfigView
    | ImagesFolderSourceConfigView,
    Field(discriminator="source_type"),
]

SourceViewAdapter: TypeAdapter[SourceView] = TypeAdapter(SourceView)


class BaseSourceConfigCreate(BaseIDNameModel):
    pass


class USBCameraSourceConfigCreate(BaseSourceConfigCreate):
    source_type: Literal[SourceType.USB_CAMERA]
    device_id: int
    codec: Annotated[str | None, StringConstraints(min_length=4, max_length=4, to_upper=True)] = None

    @property
    def config_data(self) -> USBCameraConfig:
        return USBCameraConfig(device_id=self.device_id, codec=self.codec)


class IPCameraSourceConfigCreate(BaseSourceConfigCreate):
    source_type: Literal[SourceType.IP_CAMERA]
    stream_url: str
    auth_required: bool = False

    @property
    def config_data(self) -> IPCameraConfig:
        return IPCameraConfig(stream_url=self.stream_url, auth_required=self.auth_required)


class VideoFileSourceConfigCreate(BaseSourceConfigCreate):
    source_type: Literal[SourceType.VIDEO_FILE]
    video_path: str
    loop: bool = False

    @property
    def config_data(self) -> VideoFileConfig:
        return VideoFileConfig(video_path=self.video_path, loop=self.loop)


class ImagesFolderSourceConfigCreate(BaseSourceConfigCreate):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    images_folder_path: str
    ignore_existing_images: bool

    @property
    def config_data(self) -> ImagesFolderConfig:
        return ImagesFolderConfig(
            images_folder_path=self.images_folder_path, ignore_existing_images=self.ignore_existing_images
        )


SourceCreate = Annotated[
    USBCameraSourceConfigCreate
    | IPCameraSourceConfigCreate
    | VideoFileSourceConfigCreate
    | ImagesFolderSourceConfigCreate,
    Field(discriminator="source_type"),
]

SourceCreateAdapter: TypeAdapter[SourceCreate] = TypeAdapter(SourceCreate)


class SourceMediaUploadView(BaseModel):
    """Response returned after uploading a video file for use as a 'video_file' source."""

    video_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_path": "/data/source_media/712750b2-5a82-47ee-8fba-f3dc96cb615d/sample.mp4",
            }
        }
    }


class SourceMediaDeletionView(BaseModel):
    """Response returned after deleting an unreferenced uploaded video file by its UUID."""

    deleted_video_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "deleted_video_path": "/data/source_media/712750b2-5a82-47ee-8fba-f3dc96cb615d/sample.mp4",
            }
        }
    }
