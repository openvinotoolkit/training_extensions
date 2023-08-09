"""Utils for processing of segmentation results."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Union

import cv2
import numpy as np


def get_activation_map(features: Union[np.ndarray, Iterable, int, float], normalize: bool = True):
    """Getter activation_map functions."""
    if normalize:
        min_soft_score = np.min(features)
        max_soft_score = np.max(features)
        factor = 255.0 / (max_soft_score - min_soft_score + 1e-12)

        float_act_map = factor * (features - min_soft_score)
        int_act_map = np.uint8(np.floor(float_act_map))
    else:
        int_act_map = features

    int_act_map = cv2.applyColorMap(int_act_map, cv2.COLORMAP_JET)
    int_act_map = cv2.cvtColor(int_act_map, cv2.COLOR_BGR2RGB)
    return int_act_map
