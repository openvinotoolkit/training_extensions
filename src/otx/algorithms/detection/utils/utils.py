"""Utils for OTX Detection."""
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

import colorsys
import random
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util

from otx.api.entities.annotation import Annotation
from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle

# pylint: disable=invalid-name


class ColorPalette:
    """ColorPalette class."""

    def __init__(self, n: int, rng: Optional[random.Random] = None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [
                (rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0)) for _ in range(candidates_num)
            ]
            min_distances = [self._min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [Color(*self._hsv2rgb(*hsv)) for hsv in hsv_colors]

    @staticmethod
    def _dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def _min_distance(cls, colors_set, color_candidate):
        distances = [cls._dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def _hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n: int):
        """Return item from index function ColorPalette."""
        return self.palette[n % len(self.palette)]

    def __len__(self):
        """Return length of ColorPalette."""
        return len(self.palette)


def generate_label_schema(label_names: Sequence[str], label_domain: Domain = Domain.DETECTION):
    """Generating label_schema function."""
    colors = ColorPalette(len(label_names)) if len(label_names) > 0 else []
    not_empty_labels = [
        LabelEntity(name=name, color=colors[i], domain=label_domain, id=ID(f"{i:08}"))  # type: ignore
        for i, name in enumerate(label_names)
    ]
    emptylabel = LabelEntity(
        name="Empty label",
        color=Color(42, 43, 46),
        is_empty=True,
        domain=label_domain,
        id=ID(f"{len(not_empty_labels):08}"),
    )

    label_schema = LabelSchemaEntity()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group)
    return label_schema


def get_det_model_api_configuration(
    label_schema: LabelSchemaEntity,
    task_type: TaskType,
    confidence_threshold: float,
    tiling_parameters: Any,
    use_ellipse_shapes: bool,
):
    """Get ModelAPI config."""
    omz_config = {}
    all_labels = ""
    all_label_ids = ""
    if task_type == TaskType.DETECTION:
        omz_config[("model_info", "model_type")] = "ssd"
        omz_config[("model_info", "task_type")] = "detection"
    if task_type == TaskType.INSTANCE_SEGMENTATION:
        omz_config[("model_info", "model_type")] = "MaskRCNN"
        omz_config[("model_info", "task_type")] = "instance_segmentation"
        all_labels = "otx_empty_lbl "
        all_label_ids = "None "
        if tiling_parameters.enable_tiling:
            omz_config[("model_info", "resize_type")] = "fit_to_window_letterbox"
    if task_type == TaskType.ROTATED_DETECTION:
        omz_config[("model_info", "model_type")] = "MaskRCNN"
        omz_config[("model_info", "task_type")] = "rotated_detection"
        all_labels = "otx_empty_lbl "
        all_label_ids = "None "
        if tiling_parameters.enable_tiling:
            omz_config[("model_info", "resize_type")] = "fit_to_window_letterbox"

    omz_config[("model_info", "confidence_threshold")] = str(confidence_threshold)
    omz_config[("model_info", "iou_threshold")] = str(0.5)
    omz_config[("model_info", "use_ellipse_shapes")] = str(use_ellipse_shapes)

    if tiling_parameters.enable_tiling:
        omz_config[("model_info", "tile_size")] = str(
            int(tiling_parameters.tile_size * tiling_parameters.tile_ir_scale_factor)
        )
        omz_config[("model_info", "tiles_overlap")] = str(
            tiling_parameters.tile_overlap / tiling_parameters.tile_ir_scale_factor
        )
        omz_config[("model_info", "max_pred_number")] = str(tiling_parameters.tile_max_number)

    for lbl in label_schema.get_labels(include_empty=False):
        all_labels += lbl.name.replace(" ", "_") + " "
        all_label_ids += f"{lbl.id_} "

    omz_config[("model_info", "labels")] = all_labels.strip()
    omz_config[("model_info", "label_ids")] = all_label_ids.strip()

    return omz_config


def expand_box(box: np.ndarray, scale_h: float, scale_w: float):
    """Expand the box.

    Args:
        box (np.ndarray): bounding box
        scale_h (float): scaling factor for height
        scale_w (float): scaling factor for width

    Returns:
        expanded box (np.ndarray): x1, y1, x2, y2 coordinates of the expanded box
    """
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5
    w_half *= scale_w
    h_half *= scale_h
    expanded_box = np.zeros(box.shape)
    expanded_box[0] = x_c - w_half
    expanded_box[2] = x_c + w_half
    expanded_box[1] = y_c - h_half
    expanded_box[3] = y_c + h_half
    return expanded_box


def mask_resize(box: np.ndarray, mask: np.ndarray, img_height: int, img_width: int):
    """Resize mask to the size of the bounding box.

    Args:
        box (np.ndarray): bounding box which enclosing the mask
        mask (np.ndarray): mask to be resize
        img_height (int): image height
        img_width (int): image width

    Returns:
        bit_mask (np.ndarray): full size mask
    """
    # scaling bbox to prevent up-sampling artifacts on segment borders.
    mask = np.pad(mask, ((1, 1), (1, 1)), "constant", constant_values=0)
    scale_h = mask.shape[0] / (mask.shape[0] - 2.0)
    scale_w = mask.shape[1] / (mask.shape[1] - 2.0)
    extended_box = expand_box(box, scale_h=scale_h, scale_w=scale_w).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[img_width, img_height])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[img_width, img_height])
    mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
    mask = mask.astype(np.uint8)
    bit_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    bit_mask[y0:y1, x0:x1] = mask[
        (y0 - extended_box[1]) : (y1 - extended_box[1]),
        (x0 - extended_box[0]) : (x1 - extended_box[0]),
    ]
    return bit_mask


def create_detection_shapes(
    pred_results: List[np.ndarray],
    width: int,
    height: int,
    confidence_threshold: float,
    use_ellipse_shapes: bool,
    labels: List,
):
    """Create prediction detection shapes.

    Args:
        pred_results (list(np.ndarray)): per class predicted boxes
        width (int): image width
        height (int): image height
        confidence_threshold (float): confidence threshold for filtering predictions
        use_ellipse_shapes (bool): if True, use ellipse shapes
        labels (list): dataset labels

    Returns:
        shapes: list of prediction shapes (Annotation)
    """

    shapes = []
    for label_idx, detections in enumerate(pred_results):
        for det in detections:
            probability = float(det[4])
            coords = det[:4].astype(float).copy()
            coords /= np.array([width, height, width, height], dtype=float)
            coords = np.clip(coords, 0, 1)

            if (probability < confidence_threshold) or (coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0):
                continue

            assigned_label = [ScoredLabel(labels[label_idx], probability=probability)]
            if not use_ellipse_shapes:
                shapes.append(
                    Annotation(
                        Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        labels=assigned_label,
                    )
                )
            else:
                shapes.append(
                    Annotation(
                        Ellipse(coords[0], coords[1], coords[2], coords[3]),
                        labels=assigned_label,
                    )
                )
    return shapes


def create_mask_shapes(
    pred_results: Tuple,
    width: int,
    height: int,
    confidence_threshold: float,
    use_ellipse_shapes: bool,
    labels: List,
    rotated_polygon: bool = False,
):
    """Create prediction mask shapes.

    Args:
        pred_results (tuple): tuple of predicted boxes and masks for each dataset item
        width (int): image width
        height (int): image height
        confidence_threshold (float): confidence threshold for filtering predictions
        use_ellipse_shapes (bool): if True, use ellipse shapes
        labels (list): dataset labels
        rotated_polygon (bool, optional): if True, use rotated polygons for mask shapes

    Returns:
        shapes: list of prediction shapes (Annotation)
    """
    shapes = []
    for label_idx, (boxes, masks) in enumerate(zip(*pred_results)):
        for mask, box in zip(masks, boxes):
            probability = float(box[4])
            if probability < confidence_threshold:
                continue

            assigned_label = [ScoredLabel(labels[label_idx], probability=probability)]
            if not use_ellipse_shapes:
                if isinstance(mask, dict):
                    mask = mask_util.decode(mask)

                if mask.shape[0] != height or mask.shape[1] != width:
                    # resize mask to the size of the bounding box
                    coords = box[:4].astype(float).copy()
                    mask = mask_resize(coords, mask, height, width)

                if mask.sum() == 0:
                    continue

                contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchies is None:
                    continue
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    # skip inner contours
                    if hierarchy[3] != -1 or len(contour) <= 2:
                        continue

                    if rotated_polygon:
                        box_points = cv2.boxPoints(cv2.minAreaRect(contour))
                        points = [Point(x=(point[0]) / width, y=(point[1]) / height) for point in box_points]
                    else:
                        points = [Point(x=(point[0][0]) / width, y=(point[0][1]) / height) for point in contour]

                    polygon = Polygon(points=points)
                    if cv2.contourArea(contour) > 0 and polygon.get_area() > 1e-12:
                        shapes.append(Annotation(polygon, labels=assigned_label, id=ID(f"{label_idx:08}")))
            else:
                ellipse = Ellipse((box[0]) / width, (box[1]) / height, (box[2]) / width, (box[3]) / height)
                shapes.append(Annotation(ellipse, labels=assigned_label, id=ID(f"{label_idx:08}")))
    return shapes
