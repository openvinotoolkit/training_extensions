# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Visualizer for results of prediction."""

from __future__ import annotations

import logging as log
import time
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np
from model_api.performance_metrics import put_highlighted_text

from .vis_utils import ColorPalette

if TYPE_CHECKING:
    from demo_package.streamer import BaseStreamer
    from model_api.models.utils import (
        ClassificationResult,
        DetectionResult,
        InstanceSegmentationResult,
        SegmentedObject,
    )


class BaseVisualizer:
    """Base class for visualizators."""

    def __init__(
        self,
        window_name: str | None = None,
        no_show: bool = False,
        delay: int | None = None,
        output: str = "./outputs",
    ) -> None:
        """Base class for visualizators.

        Args:
            window_name (str]): The name of the window. Defaults to None.
            no_show (bool): Flag to indicate whether to show the window. Defaults to False.
            delay (int]): The delay in seconds. Defaults to None.
            output (str]): The output directory. Defaults to "./outputs".

        Returns:
            None
        """
        self.window_name = "Window" if window_name is None else window_name

        self.delay = delay
        self.no_show = no_show
        if delay is None:
            self.delay = 1
        self.output = output

    def draw(
        self,
        frame: np.ndarray,
        predictions: NamedTuple,
    ) -> np.ndarray:
        """Draw annotations on the image.

        Args:
            frame: Input image
            predictions: Annotations to be drawn on the input image

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

    def video_delay(self, elapsed_time: float, streamer: BaseStreamer) -> None:
        """Check if video frames were inferenced faster than the original video FPS and delay visualizer if so.

        Args:
            elapsed_time (float): Time spent on frame inference
            streamer (BaseStreamer): Streamer object
        """
        if self.no_show:
            return
        if "VIDEO" in str(streamer.get_type()):
            fps_num = streamer.fps()
            orig_frame_time = 1 / fps_num
            if elapsed_time < orig_frame_time:
                time.sleep(orig_frame_time - elapsed_time)


class ClassificationVisualizer(BaseVisualizer):
    """Visualize the predicted classification labels by drawing the annotations on the input image.

    Example:
        >>> predictions = inference_model.predict(frame)
        >>> output = visualizer.draw(frame, predictions)
        >>> visualizer.show(output)
    """

    def draw(
        self,
        frame: np.ndarray,
        predictions: ClassificationResult,
    ) -> np.ndarray:
        """Draw classification annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image

        Returns:
            Output image with annotations.
        """
        predictions = predictions.top_labels
        if not any(predictions):
            log.warning("There are no predictions.")
            return frame

        class_label = predictions[0][1]
        font_scale = 0.7
        label_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][1]
        initial_labels_pos = frame.shape[0] - label_height * (int(1.5 * len(predictions)) + 1)

        if initial_labels_pos < 0:
            initial_labels_pos = label_height
            log.warning("Too much labels to display on this frame, some will be omitted")
        offset_y = initial_labels_pos

        header = "Label:     Score:"
        label_width = cv2.getTextSize(header, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
        put_highlighted_text(
            frame,
            header,
            (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            (255, 0, 0),
            2,
        )

        for idx, class_label, score in predictions:
            label = f"{idx}. {class_label}    {score:.2f}"
            label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
            offset_y += int(label_height * 1.5)
            put_highlighted_text(
                frame,
                label,
                (frame.shape[1] - label_width, offset_y),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                (255, 0, 0),
                2,
            )
        return frame


class SemanticSegmentationVisualizer(BaseVisualizer):
    """Visualize the predicted segmentation labels by drawing the annotations on the input image.

    Example:
        >>> masks = inference_model.predict(frame)
        >>> output = visualizer.draw(frame, masks)
        >>> visualizer.show(output)
    """

    def __init__(
        self,
        labels: list[str],
        window_name: str | None = None,
        no_show: bool = False,
        delay: int | None = None,
        output: str = "./outputs",
    ) -> None:
        """Semantic segmentation visualizer.

        Draws the segmentation masks on the input image.

        Parameters:
            labels (List[str]): List of labels.
            window_name (str | None): Name of the window (default is None).
            no_show (bool): Flag indicating whether to show the window (default is False).
            delay (int | None): Delay in milliseconds (default is None).
            output (str): Output path (default is "./outputs").

        Returns:
            None
        """
        super().__init__(window_name, no_show, delay, output)
        self.color_palette = ColorPalette(len(labels)).to_numpy_array()
        self.color_map = self._create_color_map()

    def _create_color_map(self) -> np.ndarray:
        classes = self.color_palette[:, ::-1]  # RGB to BGR
        color_map = np.zeros((256, 1, 3), dtype=np.uint8)
        classes_num = len(classes)
        color_map[:classes_num, 0, :] = classes
        color_map[classes_num:, 0, :] = np.random.uniform(0, 255, size=(256 - classes_num, 3))
        return color_map

    def _apply_color_map(self, input_2d_mask: np.ndarray) -> np.ndarray:
        input_3d = cv2.merge([input_2d_mask, input_2d_mask, input_2d_mask])
        return cv2.LUT(input_3d.astype(np.uint8), self.color_map)

    def draw(self, frame: np.ndarray, masks: SegmentedObject) -> np.ndarray:
        """Draw segmentation annotations on the image.

        Args:
            frame: Input image
            masks: Mask annotations to be drawn on the input image

        Returns:
            Output image with annotations.
        """
        masks = masks.resultImage
        output = self._apply_color_map(masks)
        return cv2.addWeighted(frame, 0.5, output, 0.5, 0)


class ObjectDetectionVisualizer(BaseVisualizer):
    """Visualizes object detection annotations on an input image."""

    def __init__(
        self,
        labels: list[str],
        window_name: str | None = None,
        no_show: bool = False,
        delay: int | None = None,
        output: str = "./outputs",
    ) -> None:
        """Object detection visualizer.

        Draws the object detection annotations on the input image.

        Parameters:
            labels (List[str]): The list of labels.
            window_name (str | None): The name of the window. Defaults to None.
            no_show (bool): Flag to control whether to show the window. Defaults to False.
            delay (int | None): The delay in milliseconds. Defaults to None.
            output (str): The output directory. Defaults to "./outputs".

        Returns:
            None
        """
        super().__init__(window_name, no_show, delay, output)
        self.labels = labels
        self.color_palette = ColorPalette(len(labels))

    def draw(
        self,
        frame: np.ndarray,
        predictions: DetectionResult,
    ) -> np.ndarray:
        """Draw instance segmentation annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image

        Returns:
            Output image with annotations.
        """
        for detection in predictions.objects:
            class_id = int(detection.id)
            color = self.color_palette[class_id]
            det_label = self.color_palette[class_id] if self.labels and len(self.labels) >= class_id else f"#{class_id}"
            xmin, ymin, xmax, ymax = detection.xmin, detection.ymin, detection.xmax, detection.ymax
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                frame,
                f"{det_label} {detection.score:.1%}",
                (xmin, ymin - 7),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                color,
                1,
            )

        return frame


class InstanceSegmentationVisualizer(BaseVisualizer):
    """Visualizes Instance Segmentation annotations on an input image."""

    def __init__(
        self,
        labels: list[str],
        window_name: str | None = None,
        no_show: bool = False,
        delay: int | None = None,
        output: str = "./outputs",
    ) -> None:
        """Instance segmentation visualizer.

        Draws the instance segmentation annotations on the input image.

        Args:
            labels (List[str]): The list of labels.
            window_name (str]): The name of the window. Defaults to None.
            no_show (bool): A flag to indicate whether to show the window. Defaults to False.
            delay (int]): The delay in milliseconds. Defaults to None.
            output (str]): The path to the output directory. Defaults to "./outputs".

        Returns:
            None
        """
        super().__init__(window_name, no_show, delay, output)
        self.labels = labels
        colors_num = len(labels) if labels else 80
        self.show_boxes = False
        self.show_scores = True
        self.palette = ColorPalette(colors_num)

    def draw(
        self,
        frame: np.ndarray,
        predictions: InstanceSegmentationResult,
    ) -> np.ndarray:
        """Draw the instance segmentation results on the input frame.

        Args:
            frame: np.ndarray - The input frame on which to draw the instance segmentation results.
            predictions: InstanceSegmentationResult - The instance segmentation results to be drawn.

        Returns:
            np.ndarray - The input frame with the instance segmentation results drawn on it.
        """
        result = frame.copy()
        output_objects = predictions.segmentedObjects
        bboxes = [[output.xmin, output.ymin, output.xmax, output.ymax] for output in output_objects]
        scores = [output.score for output in output_objects]
        masks = [output.mask for output in output_objects]
        label_names = [output.str_label for output in output_objects]

        result = self._overlay_masks(result, masks)
        return self._overlay_labels(result, bboxes, label_names, scores)

    def _overlay_masks(self, image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        all_contours = []

        for i, mask in enumerate(masks):
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if contours:
                all_contours.append(contours[0])

            mask_color = self.palette[i]
            cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
            cv2.bitwise_or(aggregated_colored_mask, mask_color, dst=aggregated_colored_mask, mask=mask)

        # Fill the area occupied by all instances with a colored instances mask image
        cv2.bitwise_and(segments_image, (0, 0, 0), dst=segments_image, mask=aggregated_mask)
        cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)

        cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)
        cv2.drawContours(image, all_contours, -1, (0, 0, 0))
        return image

    def _overlay_boxes(self, image: np.ndarray, boxes: list[np.ndarray], classes: list[int]) -> np.ndarray:
        for box, class_id in zip(boxes, classes):
            color = self.palette[class_id]
            top_left, bottom_right = box[:2], box[2:]
            image = cv2.rectangle(image, top_left, bottom_right, color, 2)
        return image

    def _overlay_labels(
        self,
        image: np.ndarray,
        boxes: list[np.ndarray],
        classes: list[str],
        scores: list[float],
    ) -> np.ndarray:
        template = "{}: {:.2f}" if self.show_scores else "{}"

        for box, score, label in zip(boxes, scores, classes):
            text = template.format(label, score)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(
                image,
                text,
                (box[0], box[1] + int(textsize[0] / 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return image
