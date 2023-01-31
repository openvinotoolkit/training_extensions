# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from PIL.Image import Resampling

import otx.mpa.modules.datasets.pipelines.transforms.cython_augments.pil_augment as pil_aug

PILImage = Image.Image
CvImage = np.ndarray
ImgTypes = Union[PILImage, CvImage]


class Augments:
    def _check_args_tf(kwargs):
        def _interpolation(kwargs):
            interpolation = kwargs.pop("resample", Resampling.BILINEAR)
            if isinstance(interpolation, (list, tuple)):
                return random.choice(interpolation)
            else:
                return interpolation

        kwargs["resample"] = _interpolation(kwargs)

    @staticmethod
    def autocontrast(img: PILImage, *args, **kwargs) -> PILImage:
        return ImageOps.autocontrast(img)

    @staticmethod
    def equalize(img: PILImage, *args, **kwargs) -> PILImage:
        return ImageOps.equalize(img)

    @staticmethod
    def solarize(img: PILImage, threshold: int, *args, **kwargs) -> PILImage:
        return ImageOps.solarize(img, threshold)

    @staticmethod
    def posterize(img: PILImage, bits_to_keep: int, *args, **kwargs) -> PILImage:
        if bits_to_keep >= 8:
            return img

        return ImageOps.posterize(img, bits_to_keep)

    @staticmethod
    def color(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        return ImageEnhance.Color(img).enhance(factor)

    @staticmethod
    def contrast(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        return ImageEnhance.Contrast(img).enhance(factor)

    @staticmethod
    def brightness(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        return ImageEnhance.Brightness(img).enhance(factor)

    @staticmethod
    def sharpness(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        return ImageEnhance.Sharpness(img).enhance(factor)

    @staticmethod
    def rotate(img: PILImage, degree: float, *args, **kwargs) -> PILImage:
        Augments._check_args_tf(kwargs)
        return img.rotate(degree, **kwargs)

    @staticmethod
    def shear_x(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        Augments._check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

    @staticmethod
    def shear_y(img: PILImage, factor: float, *args, **kwargs) -> PILImage:
        Augments._check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

    @staticmethod
    def translate_x_rel(img: PILImage, pct: float, *args, **kwargs) -> PILImage:
        Augments._check_args_tf(kwargs)
        pixels = pct * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

    @staticmethod
    def translate_y_rel(img: PILImage, pct: float, *args, **kwargs) -> PILImage:
        Augments._check_args_tf(kwargs)
        pixels = pct * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


class CythonAugments(Augments):
    def autocontrast(img: ImgTypes, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.autocontrast(img)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def equalize(img: ImgTypes, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.equalize(img)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def solarize(img: ImgTypes, threshold: int, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.solarize(img, threshold)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def posterize(img: ImgTypes, bits_to_keep: int, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            if bits_to_keep >= 8:
                return img

            return pil_aug.posterize(img, bits_to_keep)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def color(img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.color(img, factor)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def contrast(img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.contrast(img, factor)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def brightness(img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.brightness(img, factor)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def sharpness(img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.sharpness(img, factor)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def rotate(img: ImgTypes, degree: float, *args, **kwargs) -> ImgTypes:
        Augments._check_args_tf(kwargs)

        if Image.isImageType(img):
            return pil_aug.rotate(img, degree)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def shear_x(img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        Augments._check_args_tf(kwargs)

        if Image.isImageType(img):
            return pil_aug.shear_x(img, factor)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def shear_y(img: ImgTypes, factor: float, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.shear_y(img, factor)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def translate_x_rel(img: ImgTypes, pct: float, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.translate_x_rel(img, pct)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def translate_y_rel(img: ImgTypes, pct: float, *args, **kwargs) -> ImgTypes:
        if Image.isImageType(img):
            return pil_aug.translate_y_rel(img, pct)

        raise NotImplementedError(f"Unknown type: {type(img)}")

    def blend(src: ImgTypes, dst: CvImage, weight: float):
        assert isinstance(dst, CvImage), f"Type of dst should be numpy array, but type(dst)={type(dst)}."

        if Image.isImageType(src):
            return pil_aug.blend(src, dst, weight)

        raise NotImplementedError(f"Unknown type: {type(src)}")
