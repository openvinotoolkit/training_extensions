"""Collection of compose pipelines for segmentation task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from mmcv.utils import build_from_cfg
from mmseg.datasets.builder import PIPELINES
from scipy.ndimage import gaussian_filter


# pylint: disable=consider-using-f-string
@PIPELINES.register_module()
class ProbCompose:
    """Compose pipelines in a list and enable or disable them with the probability."""

    def __init__(self, transforms, probs):
        assert isinstance(transforms, Sequence)
        assert isinstance(probs, Sequence)
        assert len(transforms) == len(probs)
        assert all(p >= 0.0 for p in probs)

        sum_probs = float(sum(probs))
        assert sum_probs > 0.0
        norm_probs = [float(p) / sum_probs for p in probs]
        self.limits = np.cumsum([0.0] + norm_probs)

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be callable or a dict, but got {type(transform)}")

    def __call__(self, data):
        """Callback function of ProbCompose."""
        rand_value = np.random.rand()
        transform_id = np.max(np.where(rand_value > self.limits)[0])

        transform = self.transforms[transform_id]
        data = transform(data)

        return data

    def __repr__(self):
        """Repr."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"

        return format_string


@PIPELINES.register_module()
class MaskCompose:
    """Compose mask-related pipelines in a list and enable or disable them with the probability."""

    def __init__(self, transforms, prob, lambda_limits=(4, 16), keep_original=False):
        self.keep_original = keep_original
        self.prob = prob
        assert 0.0 <= self.prob <= 1.0

        assert isinstance(lambda_limits, Sequence)
        assert len(lambda_limits) == 2
        assert 0.0 < lambda_limits[0] < lambda_limits[1]
        self.lambda_limits = lambda_limits

        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be callable or a dict, but got {type(transform)}")

    @staticmethod
    def _apply_transforms(data, transforms):
        for t in transforms:
            data = t(data)
            if data is None:
                return None

        return data

    @staticmethod
    def _generate_mask(shape, lambda_limits):
        noise = np.random.randn(*shape)

        sigma = np.exp(np.log10(np.random.uniform(lambda_limits[0], lambda_limits[1])))
        soft_mask = gaussian_filter(noise, sigma=sigma)

        threshold = np.median(soft_mask)
        hard_mask = soft_mask > threshold

        return hard_mask

    @staticmethod
    def _mix_img(main_img, aux_img, mask):
        return np.where(np.expand_dims(mask, axis=2), main_img, aux_img)

    def __call__(self, data):
        """Callback function of MaskCompose."""
        main_data = self._apply_transforms(deepcopy(data), self.transforms)
        assert main_data is not None
        if not self.keep_original and np.random.rand() > self.prob:
            return main_data

        aux_data = self._apply_transforms(deepcopy(data), self.transforms)
        assert aux_data is not None

        assert main_data["img"].shape == aux_data["img"].shape

        mask = self._generate_mask(main_data["img"].shape[:2], self.lambda_limits)
        mixed_img = self._mix_img(main_data["img"], aux_data["img"], mask)

        if self.keep_original:
            main_data["aux_img"] = mixed_img
        else:
            main_data["img"] = mixed_img

        return main_data

    def __repr__(self):
        """Repr."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"

        return format_string
