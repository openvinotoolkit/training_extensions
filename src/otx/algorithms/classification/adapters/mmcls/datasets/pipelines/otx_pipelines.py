"""Collection Pipeline for classification task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import copy
from typing import Any, Dict, List

import numpy as np
from mmcls.datasets import PIPELINES
from mmcls.datasets.pipelines import Compose
from mmcv.utils.registry import build_from_cfg
from PIL import Image, ImageFilter
from torchvision import transforms as T

import otx.core.data.pipelines.load_image_from_otx_dataset as load_image_base

# TODO: refactoring to common modules
# TODO: refactoring to Sphinx style.


@PIPELINES.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""


@PIPELINES.register_module()
class RandomAppliedTrans:
    """Randomly applied transformations.

    :param transforms: List of transformations in dictionaries
    :param p: Probability, defaults to 0.5
    """

    def __init__(self, transforms: List, p: float = 0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]  # pylint: disable=invalid-name
        self.trans = T.RandomApply(t, p=p)

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

    :param sigma_min: Minimum value of sigma of gaussian filter.
    :param sigma_max: Maximum value of sigma of gaussian filter.
    """

    def __init__(self, sigma_min: float, sigma_max: float):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

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


@PIPELINES.register_module()
class PILImageToNDArray:
    """Pipeline element that converts an PIL image into numpy array.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['img']: PIL type image in data pipeline.

    Args:
        keys (list[str]): list to support multiple image converting from PIL to NDArray.
    """

    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, results):
        """Callback function of PILImageToNDArray."""
        for key in self.keys:
            img = results[key]
            img = np.asarray(img)
            results[key] = img
        return results

    def __repr__(self):
        """Repr function of PILImageToNDArray."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class PostAug:
    """Pipeline element that postaugments for mmcls.

    PostAug copies current augmented image and apply post augmentations for given keys.
    For example, if we apply PostAug(keys=dict(img_strong=strong_pipeline),
    PostAug will copy current augmented image and apply strong pipeline.
    Post augmented image will be saved at results["img_strong"].

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['img']: PIL type image in data pipeline.

    Args:
        keys (dict): keys to apply postaugmentaion. ex) dict(img_strong=strong_pipeline)
    """

    def __init__(self, keys: dict):
        self.pipelines = {key: Compose(pipeline) for key, pipeline in keys.items()}

    def __call__(self, results):
        """Callback function of PostAug."""
        for key, pipeline in self.pipelines.items():
            results[key] = pipeline(copy.deepcopy(results))["img"]
            results["img_fields"].append(key)
        return results

    def __repr__(self):
        """Repr function of PostAug."""
        repr_str = self.__class__.__name__
        return repr_str
