"""Module for defining AugMix class used for classification task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random
import re
from copy import deepcopy

import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcv.utils import ConfigDict
from PIL import Image

from otx.algorithms.common.adapters.mmcv.pipelines.transforms.augments import (
    CythonAugments,
)

_AUGMIX_TRANSFORMS_GREY = [
    "SharpnessIncreasing",  # not in paper
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]

_AUGMIX_TRANSFORMS = [
    "AutoContrast",
    "ColorIncreasing",  # not in paper
    "ContrastIncreasing",  # not in paper
    "BrightnessIncreasing",  # not in paper
    "SharpnessIncreasing",  # not in paper
    "Equalize",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]


class OpsFabric:
    """OpsFabric class."""

    def __init__(self, name, magnitude, hparams, prob=1.0):
        self.max_level = 10
        self.prob = prob
        self.hparams = hparams
        # kwargs for augment functions
        self.aug_kwargs = dict(fillcolor=hparams["img_mean"], resample=(Image.BILINEAR, Image.BICUBIC))
        self.level_to_arg = {
            "AutoContrast": None,
            "Equalize": None,
            "Rotate": self._rotate_level_to_arg,
            "PosterizeIncreasing": self._posterize_increasing_level_to_arg,
            "SolarizeIncreasing": self._solarize_increasing_level_to_arg,
            "ColorIncreasing": self._enhance_increasing_level_to_arg,
            "ContrastIncreasing": self._enhance_increasing_level_to_arg,
            "BrightnessIncreasing": self._enhance_increasing_level_to_arg,
            "SharpnessIncreasing": self._enhance_increasing_level_to_arg,
            "ShearX": self._shear_level_to_arg,
            "ShearY": self._shear_level_to_arg,
            "TranslateXRel": self._translate_rel_level_to_arg,
            "TranslateYRel": self._translate_rel_level_to_arg,
        }
        self.name_to_op = {
            "AutoContrast": CythonAugments.autocontrast,
            "Equalize": CythonAugments.equalize,
            "Rotate": CythonAugments.rotate,
            "PosterizeIncreasing": CythonAugments.posterize,
            "SolarizeIncreasing": CythonAugments.solarize,
            "ColorIncreasing": CythonAugments.color,
            "ContrastIncreasing": CythonAugments.contrast,
            "BrightnessIncreasing": CythonAugments.brightness,
            "SharpnessIncreasing": CythonAugments.sharpness,
            "ShearX": CythonAugments.shear_x,
            "ShearY": CythonAugments.shear_y,
            "TranslateXRel": CythonAugments.translate_x_rel,
            "TranslateYRel": CythonAugments.translate_y_rel,
        }
        self.aug_factory = ConfigDict(
            aug_fn=self.name_to_op[name],
            level_fn=self.level_to_arg[name],
            magnitude=magnitude,
            magnitude_std=self.hparams.get("magnitude_std", float("inf")),
        )

    @staticmethod
    def randomly_negate(value):
        """With 50% prob, negate the value."""
        # disable B311 random - used for the random sampling not for security/crypto
        return -value if random.random() > 0.5 else value  # nosec B311

    def _rotate_level_to_arg(self, level, _hparams):
        # range [-30, 30]
        level = (level / self.max_level) * 30.0
        level = self.randomly_negate(level)
        return (level,)

    def _enhance_increasing_level_to_arg(self, level, _hparams):
        # range [0.1, 1.9]
        level = (level / self.max_level) * 0.9
        level = 1.0 + self.randomly_negate(level)
        return (level,)

    def _shear_level_to_arg(self, level, _hparams):
        # range [-0.3, 0.3]
        level = (level / self.max_level) * 0.3
        level = self.randomly_negate(level)
        return (level,)

    def _translate_rel_level_to_arg(self, level, hparams):
        # default range [-0.45, 0.45]
        translate_pct = hparams.get("translate_pct", 0.45)
        level = (level / self.max_level) * translate_pct
        level = self.randomly_negate(level)
        return (level,)

    def _posterize_level_to_arg(self, level, _hparams):
        # range [0, 4], 'keep 0 up to 4 MSB of original image'
        # intensity/severity of augmentation decreases with level
        return (int((level / self.max_level) * 4),)

    def _posterize_increasing_level_to_arg(self, level, hparams):
        # range [4, 0], 'keep 4 down to 0 MSB of original image',
        # intensity/severity of augmentation increases with level
        return (4 - self._posterize_level_to_arg(level, hparams)[0],)

    def _solarize_level_to_arg(self, level, _hparams):
        # range [0, 256]
        # intensity/severity of augmentation decreases with level
        return (int((level / self.max_level) * 256),)

    def _solarize_increasing_level_to_arg(self, level, _hparams):
        # range [0, 256]
        # intensity/severity of augmentation increases with level
        return (256 - self._solarize_level_to_arg(level, _hparams)[0],)

    def __call__(self, img):
        """Call method of OpsFabric class."""
        # disable B311 random - used for the random sampling not for security/crypto
        if self.prob < 1.0 and random.random() > self.prob:  # nosec B311
            return img
        magnitude = self.aug_factory.magnitude
        magnitude_std = self.aug_factory.magnitude_std
        level_fn = self.aug_factory.level_fn
        if magnitude_std:
            if magnitude_std == float("inf"):
                # disable B311 random - used for the random sampling not for security/crypto
                magnitude = random.uniform(0, magnitude)  # nosec B311
            elif magnitude_std > 0:
                magnitude = random.gauss(magnitude, magnitude_std)
        magnitude = min(self.max_level, max(0, magnitude))  # clip to valid range
        level_args = level_fn(magnitude, self.hparams) if level_fn is not None else tuple()
        return self.aug_factory.aug_fn(img, *level_args, **self.aug_kwargs)


@PIPELINES.register_module()
class AugMixAugment:
    """AugMix Transform.

    Adapted and improved from impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781.
    """

    def __init__(self, config_str, image_mean=None, grey=False):
        self.ops, self.alpha, self.width, self.depth = self._augmix_ops(config_str, image_mean, grey=grey)

    def _apply_basic(self, img, mixing_weights, m):  # pylint: disable=invalid-name
        # This is a literal adaptation of the paper/official implementation without normalizations and
        # PIL <-> Numpy conversions between every op. It is still quite CPU compute heavy compared to the
        # typical augmentation transforms, could use a GPU / Kornia implementation.
        mixed = (1 - m) * np.array(img, dtype=np.float32)
        for mix_weight in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = deepcopy(img)
            for op in ops:  # pylint: disable=invalid-name
                img_aug = op(img_aug)
            CythonAugments.blend(img_aug, mixed, mix_weight * m)
        np.clip(mixed, 0, 255.0, out=mixed)
        return Image.fromarray(mixed.astype(np.uint8))

    def _augmix_ops(self, config_str, image_mean=None, translate_const=250, grey=False):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]  # imagenet mean
        aug_params = ConfigDict(magnitude=3, width=3, depth=-1, alpha=1.0, p=1.0)
        hparams = dict(
            translate_const=translate_const,
            img_mean=tuple(int(c * 256) for c in image_mean),
            magnitude_std=float("inf"),
        )
        config = config_str.split("-")
        assert config[0] == "augmix"
        config = config[1:]
        for cfg in config:
            cfgs = re.split(r"(\d.*)", cfg)
            if len(cfgs) < 2:
                continue
            key, val = cfgs[:2]
            if key == "mstd":
                hparams.setdefault("magnitude_std", float(val))
            elif key == "m":
                aug_params.magnitude = int(val)
            elif key == "w":
                aug_params.width = int(val)
            elif key == "d":
                aug_params.depth = int(val)
            elif key == "a":
                aug_params.alpha = float(val)
            elif key == "p":
                aug_params.p = float(val)
            else:
                assert False, "Unknown AugMix config section"
        aug_politics = _AUGMIX_TRANSFORMS_GREY if grey else _AUGMIX_TRANSFORMS
        return (
            [OpsFabric(name, aug_params.magnitude, hparams, aug_params.p) for name in aug_politics],
            aug_params.alpha,
            aug_params.width,
            aug_params.depth,
        )

    def __call__(self, results):
        """Call function applies augmix on image."""
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if not Image.isImageType(img):
                img = Image.fromarray(img)
            mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
            m = np.float32(np.random.beta(self.alpha, self.alpha))  # pylint: disable=invalid-name
            mixed = self._apply_basic(img, mixing_weights, m)
            results["augmix"] = True
            results[key] = mixed
        return results
