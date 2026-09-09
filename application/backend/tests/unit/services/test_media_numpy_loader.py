# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.models.media import Media
from app.services.media_numpy_loader import BinaryNotFoundError, MediaNumpyLoader
from app.services.media_service import MediaService

# Distinct per-channel values so that any channel reordering (or lack thereof) is detectable.
BLUE, GREEN, RED, ALPHA = 10, 20, 30, 40


class TestMediaNumpyLoader:
    @pytest.fixture
    def fxt_media_service(self):
        return MagicMock(spec=MediaService)

    @pytest.fixture
    def fxt_loader(self, fxt_media_service):
        return MediaNumpyLoader(media_service=fxt_media_service)

    @pytest.fixture
    def fxt_media(self):
        media = MagicMock(spec=Media)
        media.id = uuid4()
        return media

    @staticmethod
    def _write(tmp_path: Path, array: np.ndarray, suffix: str = ".png") -> Path:
        path = tmp_path / f"media{suffix}"
        assert cv2.imwrite(str(path), array)
        return path

    def test_load_bgra_media_is_converted_to_rgba(self, fxt_loader, fxt_media_service, fxt_media, tmp_path):
        # A 4-channel image is stored on disk as BGRA by OpenCV.
        bgra = np.full((2, 3, 4), (BLUE, GREEN, RED, ALPHA), dtype=np.uint8)
        fxt_media_service.get_media_binary_path.return_value = self._write(tmp_path, bgra)

        result = fxt_loader.load_media_binary(project_id=uuid4(), media=fxt_media)

        assert result.shape == (2, 3, 4)
        assert result.dtype == np.uint8
        # Channels must come back as RGBA, i.e. red and blue swapped relative to the file.
        np.testing.assert_array_equal(np.unique(result.reshape(-1, 4), axis=0), [[RED, GREEN, BLUE, ALPHA]])

    def test_load_bgr_media_is_converted_to_rgb(self, fxt_loader, fxt_media_service, fxt_media, tmp_path):
        bgr = np.full((2, 3, 3), (BLUE, GREEN, RED), dtype=np.uint8)
        fxt_media_service.get_media_binary_path.return_value = self._write(tmp_path, bgr)

        result = fxt_loader.load_media_binary(project_id=uuid4(), media=fxt_media)

        assert result.shape == (2, 3, 3)
        np.testing.assert_array_equal(np.unique(result.reshape(-1, 3), axis=0), [[RED, GREEN, BLUE]])

    def test_load_grayscale_media_gets_channel_dimension(self, fxt_loader, fxt_media_service, fxt_media, tmp_path):
        gray = np.arange(6, dtype=np.uint8).reshape(2, 3)
        fxt_media_service.get_media_binary_path.return_value = self._write(tmp_path, gray)

        result = fxt_loader.load_media_binary(project_id=uuid4(), media=fxt_media)

        assert result.shape == (2, 3, 1)
        np.testing.assert_array_equal(result[..., 0], gray)

    def test_load_high_bit_depth_media_preserves_dtype(self, fxt_loader, fxt_media_service, fxt_media, tmp_path):
        gray16 = np.array([[0, 30000], [60000, 65535]], dtype=np.uint16)
        fxt_media_service.get_media_binary_path.return_value = self._write(tmp_path, gray16)

        result = fxt_loader.load_media_binary(project_id=uuid4(), media=fxt_media)

        assert result.shape == (2, 2, 1)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result[..., 0], gray16)

    def test_load_missing_binary_raises(self, fxt_loader, fxt_media_service, fxt_media, tmp_path):
        fxt_media_service.get_media_binary_path.return_value = tmp_path / "does-not-exist.png"

        with pytest.raises(BinaryNotFoundError):
            fxt_loader.load_media_binary(project_id=uuid4(), media=fxt_media)
