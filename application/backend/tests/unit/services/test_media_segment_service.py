# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from safetensors.numpy import load as safetensors_load
from sqlalchemy.orm import Session

from app.models import Project, Video
from app.models.media import Image, MediaType, NotAnnotatedVideoFrame, VideoFrame
from app.models.system import DeviceInfo, DeviceType
from app.services.media_numpy_loader import MediaNumpyLoader
from app.services.media_service import MediaService
from app.services.sam.media_segment_service import (
    SAM_EMBEDDING_MODEL_VERSION,
    MediaSegmentService,
    _get_encoder_plugin_config,
)


def _read_safetensors_metadata(blob: bytes) -> dict[str, str]:
    """Extract the ``__metadata__`` slot from a safetensors byte blob.

    ``safetensors.numpy.load`` only returns the tensors, so the header is parsed
    manually: the first 8 bytes are a little-endian u64 header length, followed by
    a JSON header that carries the ``__metadata__`` mapping.
    """
    header_len = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + header_len])
    return header.get("__metadata__", {})


class TestMediaSegmentService:
    @pytest.fixture
    def fxt_media_service(self):
        return MagicMock(spec=MediaService)

    @pytest.fixture
    def fxt_media_numpy_loader(self):
        return MagicMock(spec=MediaNumpyLoader)

    @pytest.fixture
    def fxt_media_segment_service(self, fxt_media_service, fxt_media_numpy_loader):
        return MediaSegmentService(
            media_service=fxt_media_service,
            media_numpy_loader=fxt_media_numpy_loader,
            model_xml_path=Path("/tmp/mobile_sam.encoder.xml"),
            ov_cache_path=None,
            db_session=MagicMock(spec=Session),
        )

    @pytest.mark.parametrize(
        "height, width, expected_new_height, expected_new_width",
        [
            (1024, 1024, 1024, 1024),  # already square, longest side already 1024
            (768, 1024, 768, 1024),  # landscape, width is the longest side
            (1024, 768, 1024, 768),  # portrait, height is the longest side
            (2048, 1024, 1024, 512),  # downscale by 0.5
            (500, 500, 1024, 1024),  # upscale a small square
            (100, 300, 341, 1024),  # rounding: 100 * (1024/300) + 0.5 -> 341
        ],
    )
    def test_compute_resize_metadata(self, height, width, expected_new_height, expected_new_width):
        metadata = MediaSegmentService._compute_resize_metadata(original_height=height, original_width=width)

        assert metadata == {
            "original_width": str(width),
            "original_height": str(height),
            "new_width": str(expected_new_width),
            "new_height": str(expected_new_height),
        }
        # Every value must be a string so it can go into safetensors' __metadata__ slot.
        assert all(isinstance(value, str) for value in metadata.values())

    def test_serialize_roundtrip(self):
        embeddings = np.random.rand(1, 256, 64, 64).astype(np.float32)
        resize_metadata = {
            "original_width": "1024",
            "original_height": "768",
            "new_width": "1024",
            "new_height": "768",
        }

        blob = MediaSegmentService._serialize(embeddings=embeddings, resize_metadata=resize_metadata)

        tensors = safetensors_load(blob)
        np.testing.assert_array_equal(tensors["image_embeddings"], embeddings)
        assert tensors["image_embeddings"].dtype == np.float32

        metadata = _read_safetensors_metadata(blob)
        assert metadata == {**resize_metadata, "model_version": SAM_EMBEDDING_MODEL_VERSION}

    def test_serialize_handles_non_contiguous_array(self):
        # A transposed view is not C-contiguous; _serialize must normalize it.
        embeddings = np.random.rand(64, 256).astype(np.float32).T
        assert not embeddings.flags["C_CONTIGUOUS"]

        blob = MediaSegmentService._serialize(embeddings=embeddings, resize_metadata={})

        tensors = safetensors_load(blob)
        np.testing.assert_array_equal(tensors["image_embeddings"], embeddings)

    def test_load_model_pins_precision_per_platform(self, fxt_media_service, fxt_media_numpy_loader, tmp_path):
        service = MediaSegmentService(
            media_service=fxt_media_service,
            media_numpy_loader=fxt_media_numpy_loader,
            model_xml_path=Path("/tmp/mobile_sam.encoder.xml"),
            ov_cache_path=tmp_path,
            db_session=MagicMock(spec=Session),
        )
        core = MagicMock()

        with (
            patch("model_api.adapters.create_core", return_value=core),
            patch("model_api.adapters.OpenvinoAdapter") as mock_adapter,
            patch("model_api.models.Model.create_model"),
        ):
            service._load_model(device="CPU")

        core.set_property.assert_any_call({"CACHE_DIR": str(tmp_path)})
        assert mock_adapter.call_args.kwargs["plugin_config"] == _get_encoder_plugin_config()

    @pytest.mark.parametrize(
        "machine, expected_config",
        [
            ("arm64", {"INFERENCE_PRECISION_HINT": "f32"}),
            ("aarch64", {"INFERENCE_PRECISION_HINT": "f32"}),
            ("x86_64", {}),
        ],
    )
    def test_get_encoder_plugin_config(self, machine, expected_config):
        with (
            patch("app.services.sam.media_segment_service.platform.machine", return_value=machine),
        ):
            assert _get_encoder_plugin_config() == expected_config

    def test_encode_media_image(self, fxt_media_segment_service, fxt_media_numpy_loader):
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=uuid4(), type=MediaType.IMAGE)
        device = DeviceInfo.cpu()

        media_binary = np.zeros((768, 1024, 3), dtype=np.uint8)
        fxt_media_numpy_loader.load_media_binary.return_value = media_binary

        embeddings = np.random.rand(1, 256, 64, 64).astype(np.float32)
        model = MagicMock(return_value=embeddings)

        with patch.object(fxt_media_segment_service, "_load_model", return_value=model) as mock_load_model:
            result = fxt_media_segment_service.encode_media(project=project, media=image, device=device)

        fxt_media_numpy_loader.load_media_binary.assert_called_once_with(project_id=project.id, media=image)
        mock_load_model.assert_called_once_with(device="CPU")
        model.assert_called_once_with(media_binary)

        tensors = safetensors_load(result)
        np.testing.assert_array_equal(tensors["image_embeddings"], embeddings)
        metadata = _read_safetensors_metadata(result)
        assert metadata == {
            **MediaSegmentService._compute_resize_metadata(original_height=768, original_width=1024),
            "model_version": SAM_EMBEDDING_MODEL_VERSION,
        }

    def test_encode_media_video_frame(self, fxt_media_segment_service, fxt_media_numpy_loader):
        project = MagicMock(spec=Project, id=uuid4())
        video_frame = MagicMock(spec=VideoFrame, id=uuid4(), type=MediaType.VIDEO_FRAME)
        device = DeviceInfo.cpu()

        media_binary = np.zeros((1024, 512, 3), dtype=np.uint8)
        fxt_media_numpy_loader.load_media_binary.return_value = media_binary

        embeddings = np.random.rand(1, 256, 64, 64).astype(np.float32)
        model = MagicMock(return_value=embeddings)

        with patch.object(fxt_media_segment_service, "_load_model", return_value=model):
            result = fxt_media_segment_service.encode_media(project=project, media=video_frame, device=device)

        fxt_media_numpy_loader.load_media_binary.assert_called_once_with(project_id=project.id, media=video_frame)
        metadata = _read_safetensors_metadata(result)
        assert metadata["original_height"] == "1024"
        assert metadata["original_width"] == "512"

    def test_encode_media_not_annotated_video_frame(self, fxt_media_segment_service, fxt_media_service):
        project = MagicMock(spec=Project, id=uuid4())
        video = MagicMock(spec=Video, id=uuid4())
        media = MagicMock(spec=NotAnnotatedVideoFrame, video=video, frame_index=7)
        device = DeviceInfo.cpu()

        media_binary = np.zeros((720, 1280, 3), dtype=np.uint8)
        fxt_media_service.get_frame_binaries.return_value = {7: media_binary}

        embeddings = np.random.rand(1, 256, 64, 64).astype(np.float32)
        model = MagicMock(return_value=embeddings)

        with patch.object(fxt_media_segment_service, "_load_model", return_value=model):
            result = fxt_media_segment_service.encode_media(project=project, media=media, device=device)

        fxt_media_service.get_frame_binaries.assert_called_once_with(project=project, video=video, frame_indexes=[7])
        model.assert_called_once_with(media_binary)

        tensors = safetensors_load(result)
        np.testing.assert_array_equal(tensors["image_embeddings"], embeddings)
        metadata = _read_safetensors_metadata(result)
        assert metadata["original_height"] == "720"
        assert metadata["original_width"] == "1280"

    def test_encode_media_video_raises(self, fxt_media_segment_service):
        project = MagicMock(spec=Project, id=uuid4())
        video = MagicMock(spec=Video, id=uuid4(), type=MediaType.VIDEO)
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        with pytest.raises(ValueError, match="Video media type is not supported"):
            fxt_media_segment_service.encode_media(project=project, media=video, device=device)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            pytest.param((256, 256, 1), np.uint16, id="16bit_grayscale"),
            pytest.param((256, 256), np.uint16, id="16bit_grayscale_2d"),
            pytest.param((256, 256, 3), np.uint16, id="16bit_rgb"),
            pytest.param((256, 256, 1), np.uint8, id="8bit_grayscale"),
            pytest.param((256, 256, 4), np.uint8, id="8bit_rgba"),
        ],
    )
    def test_encode_media_converts_input_to_uint8_rgb(
        self, fxt_media_segment_service, fxt_media_numpy_loader, shape, dtype
    ):
        """The SAM encoder IR only accepts (1, H, W, 3) uint8, so any decoded media must be converted.

        Regression test: 16-bit / grayscale images used to be forwarded as-is and made
        OpenVINO raise a shape mismatch (``{1, ?, ?, 3}`` vs ``{1, 256, 256, 1}``).
        """
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=uuid4(), type=MediaType.IMAGE)
        device = DeviceInfo.cpu()

        rng = np.random.default_rng(0)
        media_binary = (rng.random(shape) * np.iinfo(dtype).max).astype(dtype)
        fxt_media_numpy_loader.load_media_binary.return_value = media_binary

        model = MagicMock(return_value=np.random.rand(1, 256, 64, 64).astype(np.float32))
        with patch.object(fxt_media_segment_service, "_load_model", return_value=model):
            result = fxt_media_segment_service.encode_media(project=project, media=image, device=device)

        model_input = model.call_args.args[0]
        assert model_input.dtype == np.uint8
        assert model_input.shape == (256, 256, 3)

        # The resize metadata still reflects the original media dimensions.
        metadata = _read_safetensors_metadata(result)
        assert metadata["original_height"] == "256"
        assert metadata["original_width"] == "256"
