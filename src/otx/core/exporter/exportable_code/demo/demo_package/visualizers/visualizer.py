"""Visualizer for results of prediction."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..streamer import BaseStreamer


class BaseVisualizer:
    """Interface for converter."""

    def __init__(
        self,
        window_name: Optional[str] = None,
        no_show: bool = False,
        delay: Optional[int] = None,
        output: Optional[str] = None,
    ) -> None:
        self.window_name = "Window" if window_name is None else window_name

        self.delay = delay
        self.no_show = no_show
        if delay is None:
            self.delay = 1
        self.output = output

    def draw(
        self,
        image: np.ndarray,
        predictions: list,
        meta: dict,
    ) -> np.ndarray:
        """Draw annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image
            metadata: Metadata is needed to render

        Returns:
            Output image with annotations.
        """
        raise NotImplementedError

    def show(self, image: np.ndarray) -> None:
        """Show result image.

        Args:
            image (np.ndarray): Image to be shown.
        """
        if not self.no_show:
            cv2.imshow(self.window_name, image)

    def is_quit(self) -> bool:
        """Check user wish to quit."""
        if self.no_show:
            return False

        return ord("q") == cv2.waitKey(self.delay)

    def video_delay(self, elapsed_time: float, streamer: BaseStreamer):
        """Check if video frames were inferenced faster than the original video FPS and delay visualizer if so.

        Args:
            elapsed_time (float): Time spent on frame inference
            streamer (BaseStreamer): Streamer object
        """
        if self.no_show:
            return
        if "VIDEO" in str(streamer.get_type()):
            orig_frame_time = 1 / streamer.fps()
            if elapsed_time < orig_frame_time:
                time.sleep(orig_frame_time - elapsed_time)


class FakeVisualizer(BaseVisualizer):
    def draw(
        self,
        image: np.ndarray,
        predictions: list,
        meta: dict,
    ) -> np.ndarray:
        """Immitate drawing annotations.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image
            metadata: Metadata is needed to render

        Returns:
            Output same image without any annotations.
        """
        return image


class ClassificationVisualizer(BaseVisualizer):
    """Visualize the predicted classification labels by drawing the annotations on the input image.

    Example:
        >>> predictions = inference_model.predict(frame)
        >>> output = visualizer.draw(frame, predictions)
        >>> visualizer.show(output)
    """

    def draw(
        self,
        image: np.ndarray,
        predictions: list,
        meta: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image

        Returns:
            Output image with annotations.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for label in predictions:
            image = cv2.putText(image, label)

        return image
