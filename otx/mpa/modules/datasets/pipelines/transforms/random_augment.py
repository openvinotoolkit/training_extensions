# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import random

import numpy as np
import PIL
from mmcls.datasets.builder import PIPELINES

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img), None


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v), v


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v), v


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v), v


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v), v


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img, xy, color


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img), None


def Identity(img, **kwarg):
    return img, None


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v), v


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v), v


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v), v


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), v


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), v


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v), v


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), v


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), v


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


rand_augment_pool = [
    (AutoContrast, None, None),
    (Brightness, 0.9, 0.05),
    (Color, 0.9, 0.05),
    (Contrast, 0.9, 0.05),
    (Equalize, None, None),
    (Identity, None, None),
    (Posterize, 4, 4),
    (Rotate, 30, 0),
    (Sharpness, 0.9, 0.05),
    (ShearX, 0.3, 0),
    (ShearY, 0.3, 0),
    (Solarize, 256, 0),
    (TranslateX, 0.3, 0),
    (TranslateY, 0.3, 0),
]

# TODO: [Jihwan]: Can be removed by mmcls.datasets.pipeline.auto_augment Line 95 RandAugment class
@PIPELINES.register_module()
class MPARandAugment(object):
    def __init__(self, n, m, cutout=16):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.cutout = cutout
        self.augment_pool = rand_augment_pool

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if not PIL.Image.isImageType(img):
                img = PIL.Image.fromarray(results[key])
            ops = random.choices(self.augment_pool, k=self.n)
            for op, max_v, bias in ops:
                v = np.random.randint(1, self.m)
                if random.random() < 0.5:
                    img, v = op(img, v=v, max_v=max_v, bias=bias)
                    results["rand_mc_{}".format(op.__name__)] = v
            img, xy, color = CutoutAbs(img, self.cutout)
            results["CutoutAbs"] = (xy, self.cutout, color)
            results[key] = np.array(img)
        return results
