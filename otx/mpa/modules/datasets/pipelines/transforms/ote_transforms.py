# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random

from torchvision.transforms import functional as F
from PIL import Image

from mmcls.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PILToTensor(object):
    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if not Image.isImageType(img):
                img = Image.fromarray(img)
            img = F.to_tensor(img)
            results["PILToTensor"] = True
            results[key] = img
        return results


@PIPELINES.register_module()
class TensorNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img = F.normalize(img, self.mean, self.std, self.inplace)
            results["TensorNormalize"] = True
            results[key] = img
        return results


@PIPELINES.register_module()
class RandomRotate(object):
    """Random rotate
    From torchreid.data.transforms
    """

    def __init__(self, p=0.5, angle=(-5, 5), values=None, **kwargs):
        self.p = p
        self.angle = angle
        self.discrete = values is not None and len([v for v in values if v != 0]) > 0
        self.values = values

    def __call__(self, results, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if self.discrete:
                rnd_angle = float(self.values[random.randint(0, len(self.values) - 1)])
            else:
                rnd_angle = random.randint(self.angle[0], self.angle[1])
            if not Image.isImageType(img):
                img = Image.fromarray(img)

            img = F.rotate(img, rnd_angle, expand=False, center=None)
            results["RandomRotate"] = True
            results[key] = img

        return results
