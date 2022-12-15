"""Collection Pipeline for classification task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

import numpy as np
from mmcls.datasets import PIPELINES
from mmcv.utils.registry import build_from_cfg
from PIL import Image, ImageFilter
from torchvision import transforms as T

from otx.api.utils.argument_checks import check_input_parameters_type


# Temporary copy from detection_tasks
# TODO: refactoring to common modules
# TODO: refactoring to Sphinx style.
@PIPELINES.register_module()
class LoadImageFromOTXDataset:
    """Pipeline element that loads an image from a OTX Dataset on the fly.

    Can do conversion to float 32 if needed.
    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the image
        results['dataset_id']: id of the dataset to which the item belongs
        results['index']: index of the item in the dataset

    :param to_float32: optional bool, True to convert images to fp32. defaults to False
    """

    @check_input_parameters_type()
    def __init__(self, to_float32: bool = False):
        self.to_float32 = to_float32

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadImageFromOTXDataset."""
        dataset_item = results["dataset_item"]
        img = dataset_item.numpy
        shape = img.shape

        assert img.shape[0] == results["height"], f"{img.shape[0]} != {results['height']}"
        assert img.shape[1] == results["width"], f"{img.shape[1]} != {results['width']}"

        filename = f"Dataset item index {results['index']}"
        results["filename"] = filename
        results["ori_filename"] = filename
        results["img"] = img
        results["img_shape"] = shape
        results["ori_shape"] = shape
        # Set initial values for default meta_keys
        results["pad_shape"] = shape
        num_channels = 1 if len(shape) < 3 else shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        results["img_fields"] = ["img"]

        if self.to_float32:
            results["img"] = results["img"].astype(np.float32)

        return results


@PIPELINES.register_module()
class RandomAppliedTrans:
    """Randomly applied transformations.

    :param transforms: List of transformations in dictionaries
    :param p: Probability, defaults to 0.5
    """

    def __init__(self, transforms: List, p: float = 0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]  # pylint: disable=invalid-name
        self.trans = T.RandomApply(t, p=p)

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of RandomAppliedTrans.

        :param results: Inputs to be transformed.
        """
        return self.trans(results)

    def __repr__(self):
        """Set repr of RandomAppliedTrans."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class OTXColorJitter(T.ColorJitter):
    """Wrapper for ColorJitter in torchvision.transforms.

    Use this instead of mmcls's because there is no `hue` parameter in mmcls ColorJitter.
    """

    def __call__(self, results):
        """Callback function of OTXColorJitter.

        :param results: Inputs to be transformed.
        """
        results["img"] = np.array(self.forward(Image.fromarray(results["img"])))
        return results


@PIPELINES.register_module()
class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709.

    :param sigma_min:
    :param sigma_max:
    """

    @check_input_parameters_type()
    def __init__(self, sigma_min: float, sigma_max: float):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of GaussianBlur.

        :param results: Inputs to be transformed.
        """
        for key in results.get("img_fields", ["img"]):
            img = Image.fromarray(results[key])
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            results[key] = np.array(img.filter(ImageFilter.GaussianBlur(radius=sigma)))
        return results

    def __repr__(self):
        """Set repr of GaussianBlur."""
        repr_str = self.__class__.__name__
        return repr_str
