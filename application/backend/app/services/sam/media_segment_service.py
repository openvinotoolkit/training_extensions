# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from loguru import logger
from safetensors.numpy import save as safetensors_save
from sqlalchemy.orm import Session

from app.models import Image, Project, VideoFrame
from app.models.media import Media, NotAnnotatedVideoFrame, Video
from app.models.system import DeviceInfo
from app.services.base import BaseSessionManagedService
from app.services.media_numpy_loader import MediaNumpyLoader
from app.services.media_service import MediaService
from app.utils.images import to_uint8_rgb

if TYPE_CHECKING:
    from model_api.models import Model


# The SAM image encoder expects a fixed 1024x1024 input. Preprocessing rescales
# the longest side of the original image to this length while preserving the
# aspect ratio (SAM's ResizeLongestSide), so the effective (new) size is needed
# by the decoder to map prompt coordinates back onto the embedding.
SAM_INPUT_SIZE = 1024

# Identifies the encoder that produced an embedding. Bump this whenever the
# encoder model (or its preprocessing) changes so consumers can detect and
# reject embeddings that are incompatible with their decoder.
SAM_EMBEDDING_MODEL_VERSION = "mobile_sam.encoder.v1"

# The model_api SAM encoder defaults to resize_type="standard", which squashes the
# image to 1024x1024 and ignores aspect ratio. The browser SAM decoder (and SAM in
# general) assumes an aspect-preserving resize of the longest side to 1024 with the
# remaining space zero-padded at the bottom-right, so prompt coordinates map back
# with a single uniform scale from the top-left origin. Force "fit_to_window" so
# model_api rescales the longest side to 1024 while preserving aspect ratio and pads
# the shorter side at the bottom-right to reach 1024x1024; this keeps the embeddings
# aligned with what the client expects. Without it, every prompt on a non-square
# image is silently shifted.
SAM_ENCODER_CONFIGURATION = {
    "resize_type": "fit_to_window",
    "pad_value": 0,
}


# The mobile_sam encoder IR is numerically unstable in f16 on Apple Silicon (ARM):
# the graph emits the same embedding for every input, so the client decodes an empty
# mask with no error anywhere. As a workaround, we pin the inference precision to f32
# on arm64 (a.k.a. aarch64), meanwhile other platforms can use the plugin default.
def _get_encoder_plugin_config() -> dict[str, str]:
    is_arm = platform.machine() in {"arm64", "aarch64"}
    return {"INFERENCE_PRECISION_HINT": "f32"} if is_arm else {}


class MediaSegmentService(BaseSessionManagedService):
    def __init__(
        self,
        media_service: MediaService,
        media_numpy_loader: MediaNumpyLoader,
        model_xml_path: Path,
        ov_cache_path: Path | None = None,
        db_session: Session | None = None,
    ) -> None:
        super().__init__(db_session)
        self._media_service = media_service
        self._media_numpy_loader = media_numpy_loader

        self.model_xml_path = model_xml_path
        self.ov_cache_path = ov_cache_path

    def _load_model(self, device: str) -> "Model":
        from model_api.adapters import OpenvinoAdapter, create_core
        from model_api.models import Model

        logger.info("Loading SAM model '{}' on device '{}'", self.model_xml_path, device)
        core = create_core()
        core.set_property({"PERFORMANCE_HINT": "LATENCY"})
        if self.ov_cache_path is not None:
            core.set_property({"CACHE_DIR": str(self.ov_cache_path)})
        adapter = OpenvinoAdapter(
            core,
            str(self.model_xml_path),
            device=device,
            plugin_config=_get_encoder_plugin_config(),
            max_num_requests=1,
        )
        return Model.create_model(adapter, configuration=SAM_ENCODER_CONFIGURATION)

    def _unload(self, model: "Model") -> None:
        """Release all native OpenVINO resources held by the model.

        Explicitly deletes the compiled model and async queue from the adapter so that OpenVINO frees GPU/CPU memory
        immediately rather than waiting for the Python GC.
        """
        from model_api.adapters import OpenvinoAdapter

        logger.info("Unloading SAM model '{}'", self.model_xml_path)
        adapter = cast(OpenvinoAdapter, model.inference_adapter)
        if hasattr(adapter, "async_queue"):
            # Drain in-flight requests with the GIL released to avoid a GIL deadlock in the queue destructor.
            try:
                adapter.async_queue.wait_all()
            except Exception:
                logger.exception("Error while draining in-flight inference requests during unload")
            del adapter.async_queue
        if hasattr(adapter, "compiled_model"):
            del adapter.compiled_model

    @staticmethod
    def _compute_resize_metadata(original_height: int, original_width: int) -> dict[str, str]:
        """Compute the original and post-resize (aspect-preserving, longest side = 1024) dimensions.

        Values are returned as strings so they can be embedded directly in the
        safetensors ``__metadata__`` slot, which only accepts ``str`` -> ``str`` mappings.
        """
        scale = SAM_INPUT_SIZE / max(original_height, original_width)
        # SAM's ResizeLongestSide rounds with int(x + 0.5).
        new_width = int(original_width * scale + 0.5)
        new_height = int(original_height * scale + 0.5)
        return {
            "original_width": str(original_width),
            "original_height": str(original_height),
            "new_width": str(new_width),
            "new_height": str(new_height),
        }

    @staticmethod
    def _serialize(embeddings: np.ndarray, resize_metadata: dict[str, str]) -> bytes:
        """Serialize the encoder embeddings and resize metadata into a safetensors blob."""
        # safetensors requires a C-contiguous array; OpenVINO outputs already are,
        # but normalize defensively to avoid a serialization error.
        embeddings = np.ascontiguousarray(embeddings)
        # safetensors' __metadata__ only accepts str -> str, so every value is stringified.
        metadata = {**resize_metadata, "model_version": SAM_EMBEDDING_MODEL_VERSION}
        return safetensors_save({"image_embeddings": embeddings}, metadata=metadata)

    def encode_media(
        self,
        project: Project,
        media: Media | NotAnnotatedVideoFrame,
        device: DeviceInfo,
    ) -> bytes:
        if isinstance(media, Image | VideoFrame):
            media_binary = self._media_numpy_loader.load_media_binary(project_id=project.id, media=media)
        elif isinstance(media, NotAnnotatedVideoFrame):
            extracted = self._media_service.get_frame_binaries(
                project=project, video=media.video, frame_indexes=[media.frame_index]
            )
            media_binary = extracted[media.frame_index]
        elif isinstance(media, Video):
            raise ValueError("Video media type is not supported for segmentation")

        # Derive the resize metadata from the decoded binary (H, W, ...) before any channel
        # conversion; this is correct for every media type (Image, VideoFrame, extracted frame).
        original_height, original_width = int(media_binary.shape[0]), int(media_binary.shape[1])

        # The SAM encoder IR only accepts (1, H, W, 3) uint8 input, but media is decoded with its
        # original bit depth and channel count preserved (e.g. a 16-bit grayscale image decodes to
        # (H, W, 1) uint16), which makes OpenVINO reject the tensor. Normalize to 8-bit RGB using
        # the same min-max scaling the UI applies when displaying high bit depth images, so the
        # encoder sees exactly the pixels the user is prompting on.
        model_input = to_uint8_rgb(media_binary)

        model = self._load_model(device=device.as_openvino)

        media_id = (
            f"{media.video.id}_{media.frame_index}" if isinstance(media, NotAnnotatedVideoFrame) else str(media.id)
        )
        try:
            logger.debug("Performing image '{}' segmentation", media_id)
            embeddings = model(model_input)
        finally:
            # Release the native OpenVINO resources; this service loads a fresh model per request.
            self._unload(model)

        resize_metadata = self._compute_resize_metadata(original_height=original_height, original_width=original_width)
        logger.debug("Embedding resize metadata for '{}': {}", media_id, resize_metadata)

        return self._serialize(embeddings=embeddings, resize_metadata=resize_metadata)
