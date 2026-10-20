# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from app.models import Source, SourceType
from app.stream.images_folder_stream import ImagesFolderStream
from app.stream.ip_camera_stream import IPCameraStream
from app.stream.usb_camera_stream import USBCameraStream
from app.stream.video_file_stream import VideoFileStream
from app.stream.video_stream import VideoStream


class VideoStreamService:
    @staticmethod
    def get_video_stream(input_config: Source, timeout: int | None = None) -> VideoStream | None:
        video_stream: VideoStream | None = None
        try:
            match input_config.source_type:
                case SourceType.DISCONNECTED:
                    video_stream = None
                case SourceType.USB_CAMERA:
                    video_stream = USBCameraStream(
                        device_id=input_config.config_data.device_id, codec=input_config.config_data.codec
                    )
                case SourceType.IP_CAMERA:
                    video_stream = IPCameraStream(config=input_config, timeout=timeout)
                case SourceType.VIDEO_FILE:
                    video_stream = VideoFileStream(
                        video_path=input_config.config_data.video_path,
                        loop=input_config.config_data.loop,
                    )
                case SourceType.IMAGES_FOLDER:
                    video_stream = ImagesFolderStream(
                        folder_path=input_config.config_data.images_folder_path,
                        ignore_existing_images=input_config.config_data.ignore_existing_images,
                    )
                case _:
                    raise ValueError(f"Unrecognized source type: {input_config.source_type}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize video stream for source type {input_config.source_type}: {e}. "
                "Falling back to disconnected state."
            )
            video_stream = None

        if video_stream is not None:
            logger.info("Initialized video stream for source type: {}", input_config.source_type)
        else:
            logger.info("No video stream initialized")

        return video_stream
