"""Visualisation module."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


from typing import List, Tuple
from warnings import warn

import cv2
import numpy as np
from cv2 import Mat

from otx.api.entities.annotation import Annotation
from otx.api.entities.model_template import TaskType
from otx.api.entities.shapes.polygon import Polygon
from otx.api.entities.shapes.rectangle import Rectangle


def put_text_on_rect_bg(frame: Mat, message: str, position: Tuple[int, int], color=(255, 255, 0)):
    """Puts a text message on a black rectangular aread in specified position of a frame."""

    font_face = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    thickness = 1
    color_bg = (0, 0, 0)
    x, y = position
    text_size, _ = cv2.getTextSize(message, font_face, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(frame, position, (x + text_w + 1, y + text_h + 1), color_bg, -1)
    cv2.putText(
        frame,
        message,
        (x, y + text_h + font_scale - 1),
        font_face,
        font_scale,
        color,
        thickness,
    )
    return text_size


def draw_masks(frame: Mat, predictions, put_object_count: bool = False):
    """Converts predictions to masks and draw them on frame."""

    frame = frame.copy()
    height, width = frame.shape[0], frame.shape[1]
    segments_image = frame.copy()
    aggregated_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    aggregated_colored_mask = np.zeros(frame.shape, dtype=np.uint8)
    for prediction in predictions:
        if not isinstance(prediction.shape, Polygon):
            continue
        contours = np.array([[(int(p.x * width), int(p.y * height)) for p in prediction.shape.points]])
        assert len(prediction.get_labels()) == 1
        label = prediction.get_labels()[0]
        color = tuple(getattr(label.color, x) for x in ("blue", "green", "red"))
        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        cv2.drawContours(frame, contours, -1, color, 1)
        rect = cv2.boundingRect(contours[0])
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, 1)
        put_text_on_rect_bg(frame, label.name, (rect[0], rect[1]), color=color)
        cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
        cv2.bitwise_or(
            aggregated_colored_mask,
            np.asarray(color, dtype=np.uint8),
            dst=aggregated_colored_mask,
            mask=mask,
        )
    # Fill the area occupied by all instances with a colored instances mask image.
    cv2.bitwise_and(
        segments_image,
        np.zeros(3, dtype=np.uint8),
        dst=segments_image,
        mask=aggregated_mask,
    )
    cv2.bitwise_or(
        segments_image,
        aggregated_colored_mask,
        dst=segments_image,
        mask=aggregated_mask,
    )
    # Blend original image with the one, where instances are colored.
    # As a result instances masks become transparent.
    cv2.addWeighted(frame, 0.5, segments_image, 0.5, 0, dst=frame)

    if put_object_count:
        put_text_on_rect_bg(frame, f"Obj. count: {len(predictions)}", (0, 0))
    return frame


def put_labels(frame: Mat, predictions: List[Annotation]):
    """Converts predictions to text labels and puts them to the top left corner of a frame."""

    frame = frame.copy()
    assert len(predictions) == 1
    # TODO (ilya-krylov): handle multi-label classification
    assert len(predictions[0].get_labels()) == 1
    label = predictions[0].get_labels()[0]
    color = tuple(getattr(label.color, x) for x in ("blue", "green", "red"))
    put_text_on_rect_bg(frame, label.name, (0, 0), color=color)
    return frame


def draw_bounding_boxes(frame: Mat, predictions: List[Annotation], put_object_count: bool):
    """Converts predictions to bounding boxes and draws them on a frame."""

    frame = frame.copy()
    height, width = frame.shape[0], frame.shape[1]
    for prediction in predictions:
        if isinstance(prediction.shape, Rectangle):
            x1 = int(prediction.shape.x1 * width)
            x2 = int(prediction.shape.x2 * width)
            y1 = int(prediction.shape.y1 * height)
            y2 = int(prediction.shape.y2 * height)
            assert len(prediction.get_labels()) == 1
            label = prediction.get_labels()[0]
            color = tuple(getattr(label.color, x) for x in ("blue", "green", "red"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            put_text_on_rect_bg(frame, label.name, (x1, y1), color=color)
        else:
            warn(
                f"Predictions called on Annotations with shape {type(prediction.shape)}."
                "Expected shape to be of type Rectangle."
            )

    if put_object_count:
        put_text_on_rect_bg(frame, f"Obj. count: {len(predictions)}", (0, 0))
    return frame


def draw_predictions(task_type: TaskType, predictions: List[Annotation], frame: Mat, fit_to_size: Tuple[int, int]):
    """Converts predictions to visual representations depending on task type and draws them on a frame."""

    width, height = frame.shape[1], frame.shape[0]
    if fit_to_size:
        ratio_x = fit_to_size[0] / width
        ratio_y = fit_to_size[1] / height
        ratio = min(ratio_x, ratio_y)
        frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
    if task_type in {TaskType.DETECTION, TaskType.ANOMALY_DETECTION}:
        frame = draw_bounding_boxes(frame, predictions, put_object_count=True)
    elif task_type in {TaskType.CLASSIFICATION, TaskType.ANOMALY_CLASSIFICATION}:
        frame = put_labels(frame, predictions)
    elif task_type in {TaskType.INSTANCE_SEGMENTATION, TaskType.ROTATED_DETECTION}:
        frame = draw_masks(frame, predictions, put_object_count=True)
    elif task_type in {TaskType.SEGMENTATION, TaskType.ANOMALY_SEGMENTATION}:
        frame = draw_masks(frame, predictions, put_object_count=False)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return frame
