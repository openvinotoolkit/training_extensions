# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import Compose, to_tensor
from torchvision import transforms as tvt
from PIL import Image

from mmcv.utils import build_from_cfg


@PIPELINES.register_module
class ColorJitter(tvt.ColorJitter):
    def __call__(self, results):
        results["img"] = np.array(self.forward(Image.fromarray(results["img"])))
        return results


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.
    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = tvt.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class TwoCropTransform(object):
    """Generate two different cropped views of an image"""
    def __init__(self, pipeline):
        self.pipeline1 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
        self.pipeline2 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])

    def __call__(self, data):
        data1 = self.pipeline1(deepcopy(data))
        data2 = self.pipeline2(deepcopy(data))

        data = deepcopy(data1)
        data["img"] = to_tensor(
            np.ascontiguousarray(
                np.stack((data1["img"], data2["img"]), axis=0)
            )
        )
        return data
