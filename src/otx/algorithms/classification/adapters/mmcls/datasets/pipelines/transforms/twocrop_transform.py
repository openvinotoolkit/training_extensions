"""Define TwoCropTransform used for self-sl in mmclassification."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import Compose, to_tensor
from mmcv.utils import build_from_cfg


@PIPELINES.register_module()
class TwoCropTransform:
    """Generate two different cropped views of an image."""

    def __init__(self, pipeline):
        self.pipeline1 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
        self.pipeline2 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])

    def __call__(self, data):
        """Call method for TwoCropTransform class."""
        data1 = self.pipeline1(deepcopy(data))
        data2 = self.pipeline2(deepcopy(data))

        data = deepcopy(data1)
        data["img"] = to_tensor(np.ascontiguousarray(np.stack((data1["img"], data2["img"]), axis=0)))
        return data
