# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Iterable

import cv2
import numpy as np


def get_actmap(saliency_map: Union[np.ndarray, Iterable, int, float], output_res: Union[tuple, list]) -> np.ndarray:
    """ Get activation map (heatmap)  from saliency map

    Args:
        saliency_map (Union[np.ndarray, Iterable, int, float]): Saliency map with pixel values from 0-255
        output_res (Union[tuple, list]): Output resolution

    Returns:
        np.ndarray: activation map (heatmap)
    """
    if len(saliency_map.shape) == 3:
        saliency_map = saliency_map[0]

    saliency_map = cv2.resize(saliency_map, output_res)
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
    return saliency_map
