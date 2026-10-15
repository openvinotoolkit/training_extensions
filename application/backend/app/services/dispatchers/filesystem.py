# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

from app.models import FolderSinkConfig, OutputFormat

from .base import BaseDispatcher, UnavailableDispatcherError

if TYPE_CHECKING:
    from model_api.models.result import Result


class FolderDispatcher(BaseDispatcher):
    """FolderDispatcher allows outputting to a folder in the local filesystem."""

    def __init__(self, output_config: FolderSinkConfig) -> None:
        """
        Initialize the FolderDispatcher.
        Args:
            output_config: Configuration for the output destination
        """
        super().__init__(output_config=output_config)
        self.output_folder = output_config.config_data.folder_path

    def connect(self) -> None:
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def _write_image_to_file(image: np.ndarray, file_path: str) -> None:
        with open(file_path, "wb") as f:
            success, img_buf = cv2.imencode(".png", image)
            if success:
                f.write(img_buf.tobytes())
            else:
                logger.error("Failed to encode image for {}", file_path)

    @staticmethod
    def _write_predictions_to_file(predictions: str, file_path: str) -> None:
        with open(file_path, "w") as f:
            f.write(predictions)

    def test(self) -> None:
        if not os.path.isdir(self.output_folder):
            raise UnavailableDispatcherError(f"Directory not found: {self.output_folder}")
        if not os.access(self.output_folder, os.W_OK):
            raise UnavailableDispatcherError(f"Directory is not writable: {self.output_folder}")

    def _dispatch(
        self,
        original_image: np.ndarray,
        image_with_visualization: np.ndarray,
        predictions: "Result",
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")[:-3]  # up to milliseconds
        image_orig_file = os.path.join(self.output_folder, f"{timestamp}-original.jpg")
        image_viz_file = os.path.join(self.output_folder, f"{timestamp}-pred.jpg")
        pred_txt_file = os.path.join(self.output_folder, f"{timestamp}-pred.txt")

        logger.debug("Saving results to folder for timestamp '{}' to folder '{}'", timestamp, self.output_folder)

        if OutputFormat.IMAGE_ORIGINAL in self.output_formats:
            self._write_image_to_file(original_image, image_orig_file)
        if OutputFormat.IMAGE_WITH_PREDICTIONS in self.output_formats:
            self._write_image_to_file(image_with_visualization, image_viz_file)
        if OutputFormat.PREDICTIONS in self.output_formats:
            self._write_predictions_to_file(str(predictions), pred_txt_file)
