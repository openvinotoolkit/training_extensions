# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=unused-argument
"""Code in this file is adapted from.

https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
"""

import random

import numpy as np
import PIL
from mmcls.datasets.builder import PIPELINES

PARAMETER_MAX = 10


def auto_contrast(img, **kwargs):
    """Applies auto contrast to an image."""
    return PIL.ImageOps.autocontrast(img), None


def brightness(img, value, max_value, bias=0):
    """Applies brightness adjustment to an image."""
    value = _float_parameter(value, max_value) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(value), value


def color(img, value, max_value, bias=0):
    """Applies color adjustment to an image."""
    value = _float_parameter(value, max_value) + bias
    return PIL.ImageEnhance.Color(img).enhance(value), value


def contrast(img, value, max_value, bias=0):
    """Applies contrast adjustment to an image."""
    value = _float_parameter(value, max_value) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(value), value


def cutout(img, value, max_value, bias=0):
    """Applies cutout augmentation to an image."""
    if value == 0:
        return img
    value = _float_parameter(value, max_value) + bias
    value = int(value * min(img.size))
    return cutout_abs(img, value), value


def cutout_abs(img, value, **kwargs):
    """Applies cutout with absolute pixel size to an image."""
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - value / 2.0))
    y0 = int(max(0, y0 - value / 2.0))
    x1 = int(min(w, x0 + value))
    y1 = int(min(h, y0 + value))
    xy = (x0, y0, x1, y1)
    # gray
    rec_color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, rec_color)
    return img, xy, rec_color


def equalize(img, **kwargs):
    """Applies equalization to an image."""
    return PIL.ImageOps.equalize(img), None


def identity(img, **kwargs):
    """Returns the original image without any transformation."""
    return img, None


def posterize(img, value, max_value, bias=0):
    """Applies posterization to an image."""
    value = _int_parameter(value, max_value) + bias
    return PIL.ImageOps.posterize(img, value), value


def rotate(img, value, max_value, bias=0):
    """Applies rotation to an image."""
    value = _int_parameter(value, max_value) + bias
    # disable B311 random - used for the random sampling not for security/crypto
    if random.random() < 0.5:  # nosec B311
        value = -value
    return img.rotate(value), value


def sharpness(img, value, max_value, bias=0):
    """Applies Sharpness to an image."""
    value = _float_parameter(value, max_value) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(value), value


def shear_x(img, value, max_value, bias=0):
    """Applies ShearX to an image."""
    value = _float_parameter(value, max_value) + bias
    # disable B311 random - used for the random sampling not for security/crypto
    if random.random() < 0.5:  # nosec B311
        value = -value
    return img.transform(img.size, PIL.Image.AFFINE, (1, value, 0, 0, 1, 0)), value


def shear_y(img, value, max_value, bias=0):
    """Applies ShearY to an image."""
    value = _float_parameter(value, max_value) + bias
    # disable B311 random - used for the random sampling not for security/crypto
    if random.random() < 0.5:  # nosec B311
        value = -value
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, value, 1, 0)), value


def solarize(img, value, max_value, bias=0):
    """Applies Solarize to an image."""
    value = _int_parameter(value, max_value) + bias
    return PIL.ImageOps.solarize(img, 256 - value), value


def translate_x(img, value, max_value, bias=0):
    """Applies TranslateX to an image."""
    value = _float_parameter(value, max_value) + bias
    # disable B311 random - used for the random sampling not for security/crypto
    if random.random() < 0.5:  # nosec B311
        value = -value
    value = int(value * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, value, 0, 1, 0)), value


def translate_y(img, value, max_value, bias=0):
    """Applies TranslateX to an image."""
    value = _float_parameter(value, max_value) + bias
    # disable B311 random - used for the random sampling not for security/crypto
    if random.random() < 0.5:  # nosec B311
        value = -value
    value = int(value * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, value)), value


def _float_parameter(value, max_value):
    return float(value) * max_value / PARAMETER_MAX


def _int_parameter(value, max_value):
    return int(value * max_value / PARAMETER_MAX)


rand_augment_pool = [
    (auto_contrast, None, None),
    (brightness, 0.9, 0.05),
    (color, 0.9, 0.05),
    (contrast, 0.9, 0.05),
    (equalize, None, None),
    (identity, None, None),
    (posterize, 4, 4),
    (rotate, 30, 0),
    (sharpness, 0.9, 0.05),
    (shear_x, 0.3, 0),
    (shear_y, 0.3, 0),
    (solarize, 256, 0),
    (translate_x, 0.3, 0),
    (translate_y, 0.3, 0),
]


# TODO: [Jihwan]: Can be removed by mmcls.datasets.pipeline.auto_augment Line 95 RandAugment class
@PIPELINES.register_module()
class OTXRandAugment:
    """RandAugment class for OTX classification."""

    def __init__(self, num_aug, magnitude, cutout_value=16):
        assert num_aug >= 1
        assert 1 <= magnitude <= 10
        self.num_aug = num_aug
        self.magnitude = magnitude
        self.cutout_value = cutout_value
        self.augment_pool = rand_augment_pool

    def __call__(self, results):
        """Call function of OTXRandAugment class."""
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if not PIL.Image.isImageType(img):
                img = PIL.Image.fromarray(results[key])
            # disable B311 random - used for the random sampling not for security/crypto
            augs = random.choices(self.augment_pool, k=self.num_aug)  # nosec B311
            for aug, max_value, bias in augs:
                value = np.random.randint(1, self.magnitude)
                # disable B311 random - used for the random sampling not for security/crypto
                if random.random() < 0.5:  # nosec B311
                    img, value = aug(img, value=value, max_value=max_value, bias=bias)
                    results[f"rand_mc_{aug.__name__}"] = value
            img, xy, rec_color = cutout_abs(img, self.cutout_value)
            results["CutoutAbs"] = (xy, self.cutout_value, rec_color)
            results[key] = np.array(img)
        return results
