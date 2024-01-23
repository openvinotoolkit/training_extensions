"""Visualizer for results of prediction."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from ..streamer import BaseStreamer
from openvino.model_api.performance_metrics import put_highlighted_text


class BaseVisualizer:
    """Interface for converter."""

    def __init__(
        self,
        window_name: Optional[str] = None,
        no_show: bool = False,
        delay: Optional[int] = None,
        output: Optional[str] = "./outputs/model_visualization",
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
        output_transform: Optional[list] = None
    ) -> np.ndarray:
        """Draw annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image

        Returns:
            Output image with annotations.
        """
        frame = output_transform.resize(frame)
        class_label = ""
        if predictions:
            class_label = predictions[0][1]
        font_scale = 0.7
        label_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][1]
        initial_labels_pos =  frame.shape[0] - label_height * (int(1.5 * len(predictions)) + 1)

        if (initial_labels_pos < 0):
            initial_labels_pos = label_height
            log.warning('Too much labels to display on this frame, some will be omitted')
        offset_y = initial_labels_pos

        header = "Label:     Score:"
        label_width = cv2.getTextSize(header, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
        put_highlighted_text(frame, header, (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)

        for idx, class_label, score in predictions:
            label = '{}. {}    {:.2f}'.format(idx, class_label, score)
            label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
            offset_y += int(label_height * 1.5)
            predictions(frame, label, (frame.shape[1] - label_width, offset_y),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)
        return frame
