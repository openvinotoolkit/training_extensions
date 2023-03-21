"""Module for defining Augments and CythonArguments class used for classification task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=unused-argument, no-value-for-parameter
# mypy: ignore-errors

import random
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from PIL.Image import Resampling

import otx.algorithms.classification.adapters.mmcls.datasets.pipelines.transforms.cython_augments.pil_augment as pil_aug

PILImage = Image.Image
CvImage = np.ndarray
ImgTypes = Union[PILImage, CvImage]


class Augments:
    """Augments class that implements various augmentations."""

    def _check_args_tf(self, kwargs):
        def _interpolation(kwargs):
            interpolation = kwargs.pop("resample", Resampling.BILINEAR)
            if isinstance(interpolation, (list, tuple)):
                return random.choice(interpolation)
            return interpolation

        new_kwargs = {**kwargs, "resample": _interpolation(kwargs)}
        return new_kwargs

    @staticmethod
    def autocontrast(img: PILImage, *args, **kwargs) -> PILImage:
        """Apply autocontrast for an given image."""
        return ImageOps.autocontrast(img)

    @staticmethod
    def equalize(img: PILImage, *args, **kwargs) -> PILImage:
        """Apply equalize for an given image."""
        return ImageOps.equalize(img)

    @staticmethod
    def solarize(img: PILImage, threshold: int, *args, **kwargs) -> PILImage:
        """Apply solarize for an given image."""
        return ImageOps.solarize(img, threshold)

    @staticmethod
    def posterize(img: PILImage, bits_to_keep: int, *args, **kwargs) -> PILImage:
        """Apply posterize for an given image."""
        if bits_to_keep >= 8:
            return img
        return ImageOps.posterize(img, bits_to_keep)

    @staticmethod
    def color(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        """Apply color for an given image."""
        return ImageEnhance.Color(img).enhance(factor)

    @staticmethod
    def contrast(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        """Apply contrast for an given image."""
        return ImageEnhance.Contrast(img).enhance(factor)

    @staticmethod
    def brightness(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        """Apply brightness for an given image."""
        return ImageEnhance.Brightness(img).enhance(factor)

    @staticmethod
    def sharpness(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        """Apply sharpness for an given image."""
        return ImageEnhance.Sharpness(img).enhance(factor)

    @staticmethod
    def rotate(img: PILImage, degree: float, *args, **kwargs) -> PILImage:
        """Apply rotate for an given image."""
        kwargs = Augments._check_args_tf(kwargs)
        return img.rotate(degree, **kwargs)

    @staticmethod
    def shear_x(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        """Apply shear_x for an given image."""
        kwargs = Augments._check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

    @staticmethod
    def shear_y(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        """Apply shear_y for an given image."""
        kwargs = Augments._check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

    @staticmethod
    def translate_x_rel(img: PILImage, pct: float, *args, **kwargs) -> PILImage:
        """Apply translate_x_rel for an given image."""
        kwargs = Augments._check_args_tf(kwargs)
        pixels = pct * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

    @staticmethod
    def translate_y_rel(img: PILImage, pct: float, *args, **kwargs) -> PILImage:
        """Apply translate_y_rel for an given image."""
        kwargs = Augments._check_args_tf(kwargs)
        pixels = pct * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


class CythonAugments(Augments):
    """CythonAugments class that support faster augmentation with cythonizing."""

    def autocontrast(self, img: ImgTypes, *args, **kwargs) -> ImgTypes:
        """Apply autocontrast for an given image."""
        if Image.isImageType(img):
            return pil_aug.autocontrast(img)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def equalize(self, img: ImgTypes, *args, **kwargs) -> ImgTypes:
        """Apply equalize for an given image."""
        if Image.isImageType(img):
            return pil_aug.equalize(img)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def solarize(self, img: ImgTypes, threshold: int, *args, **kwargs) -> ImgTypes:
        """Apply solarize for an given image."""
        if Image.isImageType(img):
            return pil_aug.solarize(img, threshold)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def posterize(self, img: ImgTypes, bits_to_keep: int, *args, **kwargs) -> ImgTypes:
        """Apply posterize for an given image."""
        if Image.isImageType(img):
            if bits_to_keep >= 8:
                return img
            return pil_aug.posterize(img, bits_to_keep)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def color(self, img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        """Apply color for an given image."""
        if Image.isImageType(img):
            return pil_aug.color(img, factor)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def contrast(self, img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        """Apply contrast for an given image."""
        if Image.isImageType(img):
            return pil_aug.contrast(img, factor)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def brightness(self, img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        """Apply brightness for an given image."""
        if Image.isImageType(img):
            return pil_aug.brightness(img, factor)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def sharpness(self, img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        """Apply sharpness for an given image."""
        if Image.isImageType(img):
            return pil_aug.sharpness(img, factor)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def rotate(self, img: ImgTypes, degree: float, *args, **kwargs) -> ImgTypes:
        """Apply rotate for an given image."""
        Augments._check_args_tf(kwargs)

        if Image.isImageType(img):
            return pil_aug.rotate(img, degree)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def shear_x(self, img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        """Apply shear_x for an given image."""
        Augments._check_args_tf(kwargs)
        if Image.isImageType(img):
            return pil_aug.shear_x(img, factor)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def shear_y(self, img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        """Apply shear_y for an given image."""
        if Image.isImageType(img):
            return pil_aug.shear_y(img, factor)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def translate_x_rel(self, img: ImgTypes, pct: float, *args, **kwargs) -> ImgTypes:
        """Apply translate_x_rel for an given image."""
        if Image.isImageType(img):
            return pil_aug.translate_x_rel(img, pct)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def translate_y_rel(self, img: ImgTypes, pct: float, *args, **kwargs) -> ImgTypes:
        """Apply translate_y_rel for an given image."""
        if Image.isImageType(img):
            return pil_aug.translate_y_rel(img, pct)
        raise NotImplementedError(f"Unknown type: {type(img)}")

    def blend(self, src: ImgTypes, dst: CvImage, weight: float = 0.0):
        """Apply blend for an given image."""
        assert isinstance(dst, CvImage), f"Type of dst should be numpy array, but type(dst)={type(dst)}."
        if Image.isImageType(src):
            return pil_aug.blend(src, dst, weight)
        raise NotImplementedError(f"Unknown type: {type(src)}")
