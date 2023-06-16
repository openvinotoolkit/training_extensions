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
from typing import Optional, Sequence

import numpy as np

from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model_template import TaskType

import cv2
import pycocotools.mask as mask_util
from otx.api.entities.annotation import Annotation
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


def get_det_model_api_configuration(label_schema: LabelSchemaEntity, task_type: TaskType, confidence_threshold: float):
    """Get ModelAPI config."""
    omz_config = {}
    if task_type == TaskType.DETECTION:
        omz_config[("model_info", "model_type")] = "ssd"
    if task_type == TaskType.INSTANCE_SEGMENTATION:
        omz_config[("model_info", "model_type")] = "MaskRCNN"
    if task_type == TaskType.ROTATED_DETECTION:
        omz_config[("model_info", "model_type")] = "rotated_detection"

    omz_config[("model_info", "confidence_threshold")] = str(confidence_threshold)
    omz_config[("model_info", "iou_threshold")] = str(0.5)

    all_labels = ""
    for lbl in label_schema.get_labels(include_empty=False):
        all_labels += lbl.name.replace(" ", "_") + " "
    all_labels = all_labels.strip()

    omz_config[("model_info", "labels")] = all_labels

    return omz_config


def create_detection_shapes(all_results, width, height, confidence_threshold, use_ellipse_shapes, labels):
    shapes = []
    for label_idx, detections in enumerate(all_results):
        for i in range(detections.shape[0]):
            probability = float(detections[i, 4])
            coords = detections[i, :4].astype(float).copy()
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


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    raw_cls_mask = cv2.resize(raw_cls_mask.astype(np.float32), (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    return mask


def create_mask_shapes(
        all_results,
        width,
        height,
        confidence_threshold,
        use_ellipse_shapes,
        labels,
        rotated_polygon=False
        ):
    shapes = []
    for label_idx, (boxes, masks) in enumerate(zip(*all_results)):
        for mask, box in zip(masks, boxes):
            probability = float(box[4])
            if probability < confidence_threshold:
                continue

            assigned_label = [ScoredLabel(labels[label_idx], probability=probability)]
            if not use_ellipse_shapes:
                if isinstance(mask, dict):
                    mask = mask_util.decode(mask)
                coords = box[:4].astype(float).copy()
                left, top = coords[:2]
                mask = segm_postprocess(coords, mask)

                contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchies is None:
                    continue
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    # skip inner contours
                    if hierarchy[3] != -1:
                        continue
                    if len(contour) <= 2:
                        continue

                    if rotated_polygon:
                        box_points = cv2.boxPoints(cv2.minAreaRect(contour))
                        points = [Point(x=(point[0] + left) / width, y=(point[1] + top) / height) for point in box_points]
                    else:
                        points = [Point(x=(point[0][0] + left) / width, y=(point[0][1] + top) / height) for point in contour]

                    polygon = Polygon(points=points)
                    if cv2.contourArea(contour) > 0 and polygon.get_area() > 1e-12:
                        shapes.append(Annotation(polygon, labels=assigned_label, id=ID(f"{label_idx:08}")))
            else:
                ellipse = Ellipse((box[0] + left) / width, (box[1] + top) / height, (box[2] + left) / width, (box[3] + top) / height)
                shapes.append(Annotation(ellipse, labels=assigned_label, id=ID(f"{label_idx:08}")))
    return shapes
