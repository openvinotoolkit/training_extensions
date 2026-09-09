# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from uuid import UUID

import cv2
import numpy as np

from app.models.media import Media
from app.services.media_service import MediaService


class BinaryNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class MediaNumpyLoader:
    def __init__(
        self,
        media_service: MediaService,
    ) -> None:
        self._media_service = media_service

    def load_media_binary(self, project_id: UUID, media: Media) -> np.ndarray:
        """Decode a media binary into a numpy array, preserving bit depth and channel count.

        The result is always 3D but otherwise mirrors the source file: it can be
        ``(H, W, 1)`` grayscale, ``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA, with a dtype
        of ``uint8``, ``uint16``, … depending on the source file. Colour data is
        guaranteed to be in RGB(A) order: OpenCV decodes it as BGR(A), so both 3- and
        4-channel data is converted here. Grayscale has no channel order to speak of
        and is passed through unchanged.

        Args:
            project_id: ID of the project the media belongs to.
            media: Media whose binary should be decoded.

        Returns:
            A 3D array of shape ``(H, W, C)`` with ``C`` in ``{1, 3, 4}``; colour data
            (``C`` of 3 or 4) is in RGB(A) order.

        Raises:
            BinaryNotFoundError: If the binary is missing or cannot be decoded.
        """
        binary_path = self._media_service.get_media_binary_path(project_id=project_id, media=media)
        # IMREAD_UNCHANGED preserves the original bit depth (e.g. 16-bit PNG/TIFF images).
        binary_data = cv2.imread(str(binary_path), cv2.IMREAD_UNCHANGED)
        if binary_data is None:
            raise BinaryNotFoundError(f"Media {str(media.id)} binary cannot be found")
        # Add explicit channel dimension for 2D grayscale: (H, W) → (H, W, 1)
        if binary_data.ndim == 2:
            binary_data = binary_data[..., np.newaxis]
        # OpenCV decodes colour images as BGR(A); convert both to RGB(A) so consumers get a
        # consistent channel order. Grayscale (1 channel) is passed through unchanged.
        channels = binary_data.shape[-1]
        if channels == 3:
            binary_data = cv2.cvtColor(binary_data, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            binary_data = cv2.cvtColor(binary_data, cv2.COLOR_BGRA2RGBA)
        return binary_data
