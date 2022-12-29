# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random
import re
import numpy as np

from PIL import Image, ImageOps, ImageEnhance

from mmcls.datasets.builder import PIPELINES

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
    def __init__(self, name, magnitude, hparams, prob=1.0):
        self.max_level = 10
        self.prob = prob
        self.hparams = hparams
        # kwargs for augment functions
        self.aug_kwargs = dict(
            fillcolor=hparams["img_mean"], resample=(Image.BILINEAR, Image.BICUBIC)
        )
        self.LEVEL_TO_ARG = {
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
        self.NAME_TO_OP = {
            "AutoContrast": self.auto_contrast,
            "Equalize": self.equalize,
            "Rotate": self.rotate,
            "PosterizeIncreasing": self.posterize,
            "SolarizeIncreasing": self.solarize,
            "ColorIncreasing": self.color,
            "ContrastIncreasing": self.contrast,
            "BrightnessIncreasing": self.brightness,
            "SharpnessIncreasing": self.sharpness,
            "ShearX": self.shear_x,
            "ShearY": self.shear_y,
            "TranslateXRel": self.translate_x_rel,
            "TranslateYRel": self.translate_y_rel,
        }
        self.aug_fn = self.NAME_TO_OP[name]
        self.level_fn = self.LEVEL_TO_ARG[name]
        self.magnitude = magnitude
        self.magnitude_std = self.hparams.get("magnitude_std", float("inf"))

    @staticmethod
    def check_args_tf(kwargs):
        def _interpolation(kwargs):
            interpolation = kwargs.pop("resample", Image.BILINEAR)
            if isinstance(interpolation, (list, tuple)):
                return random.choice(interpolation)
            else:
                return interpolation

        kwargs["resample"] = _interpolation(kwargs)

    @staticmethod
    def auto_contrast(img, **__):
        return ImageOps.autocontrast(img)

    @staticmethod
    def equalize(img, **__):
        return ImageOps.equalize(img)

    @staticmethod
    def solarize(img, thresh, **__):
        return ImageOps.solarize(img, thresh)

    @staticmethod
    def posterize(img, bits_to_keep, **__):
        if bits_to_keep >= 8:
            return img
        return ImageOps.posterize(img, bits_to_keep)

    @staticmethod
    def contrast(img, factor, **__):
        return ImageEnhance.Contrast(img).enhance(factor)

    @staticmethod
    def color(img, factor, **__):
        return ImageEnhance.Color(img).enhance(factor)

    @staticmethod
    def brightness(img, factor, **__):
        return ImageEnhance.Brightness(img).enhance(factor)

    @staticmethod
    def sharpness(img, factor, **__):
        return ImageEnhance.Sharpness(img).enhance(factor)

    @staticmethod
    def randomly_negate(v):
        """With 50% prob, negate the value"""
        return -v if random.random() > 0.5 else v

    def shear_x(self, img, factor, **kwargs):
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

    def shear_y(self, img, factor, **kwargs):
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

    def translate_x_rel(self, img, pct, **kwargs):
        pixels = pct * img.size[0]
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

    def translate_y_rel(self, img, pct, **kwargs):
        pixels = pct * img.size[1]
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

    def rotate(self, img, degrees, **kwargs):
        self.check_args_tf(kwargs)
        return img.rotate(degrees, **kwargs)

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
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std:
            if self.magnitude_std == float("inf"):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(self.max_level, max(0, magnitude))  # clip to valid range
        level_args = (
            self.level_fn(magnitude, self.hparams)
            if self.level_fn is not None
            else tuple()
        )
        return self.aug_fn(img, *level_args, **self.aug_kwargs)


@PIPELINES.register_module()
class AugMixAugment(object):
    """AugMix Transform
    Adapted and improved from impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    """

    def __init__(self, config_str, image_mean=None, grey=False, **kwargs):
        self.ops, self.alpha, self.width, self.depth = self._augmix_ops(
            config_str, image_mean, grey=grey
        )

    def _apply_basic(self, img, mixing_weights, m):
        # This is a literal adaptation of the paper/official implementation without normalizations and
        # PIL <-> Numpy conversions between every op. It is still quite CPU compute heavy compared to the
        # typical augmentation transforms, could use a GPU / Kornia implementation.
        img_shape = img.size[0], img.size[1], len(img.getbands())
        mixed = np.zeros(img_shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            mixed += mw * np.asarray(img_aug, dtype=np.float32)
        np.clip(mixed, 0, 255.0, out=mixed)
        mixed = Image.fromarray(mixed.astype(np.uint8))
        return Image.blend(img, mixed, m)

    def _augmix_ops(self, config_str, image_mean=None, translate_const=250, grey=False):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]  # imagenet mean
        magnitude = 3
        width = 3
        depth = -1
        alpha = 1.0
        p = 1.0
        hparams = dict(
            translate_const=translate_const,
            img_mean=tuple([int(c * 256) for c in image_mean]),
            magnitude_std=float("inf"),
        )
        config = config_str.split("-")
        assert config[0] == "augmix"
        config = config[1:]
        for c in config:
            cs = re.split(r"(\d.*)", c)
            if len(cs) < 2:
                continue
            key, val = cs[:2]
            if key == "mstd":
                hparams.setdefault("magnitude_std", float(val))
            elif key == "m":
                magnitude = int(val)
            elif key == "w":
                width = int(val)
            elif key == "d":
                depth = int(val)
            elif key == "a":
                alpha = float(val)
            elif key == "p":
                p = float(val)
            else:
                assert False, "Unknown AugMix config section"
        aug_politics = _AUGMIX_TRANSFORMS_GREY if grey else _AUGMIX_TRANSFORMS
        return (
            [OpsFabric(name, magnitude, hparams, p) for name in aug_politics],
            alpha,
            width,
            depth,
        )

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if not Image.isImageType(img):
                img = Image.fromarray(img)
            mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
            m = np.float32(np.random.beta(self.alpha, self.alpha))
            mixed = self._apply_basic(img, mixing_weights, m)
            results["augmix"] = True
            results[key] = mixed
        return results
