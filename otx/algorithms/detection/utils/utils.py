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
from otx.api.utils.argument_checks import check_input_parameters_type

# pylint: disable=invalid-name


class ColorPalette:
    """ColorPalette class."""

    @check_input_parameters_type()
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

    @check_input_parameters_type()
    def __getitem__(self, n: int):
        """Return item from index function ColorPalette."""
        return self.palette[n % len(self.palette)]

    def __len__(self):
        """Return length of ColorPalette."""
        return len(self.palette)


@check_input_parameters_type()
def generate_label_schema(label_names: Sequence[str], label_domain: Domain = Domain.DETECTION):
    """Generating label_schema function."""
    colors = ColorPalette(len(label_names)) if len(label_names) > 0 else []
    not_empty_labels = [
        LabelEntity(name=name, color=colors[i], domain=label_domain, id=ID(f"{i:08}"))
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
