"""Collection of transfrom pipelines for segmentation task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Any, Dict, List

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmcv.utils import build_from_cfg
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import Compose
from mmseg.datasets.pipelines.formatting import to_tensor
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


@PIPELINES.register_module(force=True)
class Normalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        for target in ["img", "ul_w_img", "aux_img"]:
            if target in results:
                results[target] = mmcv.imnormalize(results[target], self.mean, self.std, self.to_rgb)
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        """Repr."""
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb=" f"{self.to_rgb})"
        return repr_str


@PIPELINES.register_module(force=True)
class DefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        for target in ["img", "ul_w_img", "aux_img"]:
            if target not in results:
                continue

            img = results[target]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            if len(img.shape) == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)
            elif len(img.shape) == 4:
                # for selfsl or supcon
                img = np.ascontiguousarray(img.transpose(0, 3, 1, 2)).astype(np.float32)
            else:
                raise ValueError(f"img.shape={img.shape} is not supported.")

            results[target] = DC(to_tensor(img), stack=True)

        for trg_name in ["gt_semantic_seg", "gt_class_borders", "pixel_weights"]:
            if trg_name not in results:
                continue

            out_type = np.float32 if trg_name == "pixel_weights" else np.int64
            results[trg_name] = DC(to_tensor(results[trg_name][None, ...].astype(out_type)), stack=True)

        return results

    def __repr__(self):
        """Repr."""
        return self.__class__.__name__


@PIPELINES.register_module()
class BranchImage:
    """Branch images by copying with name of key.

    Args:
        key_map (dict): keys to name each image.
    """

    def __init__(self, key_map):
        self.key_map = key_map

    def __call__(self, results):
        """Call function to branch images in img_fields in results.

        Args:
            results (dict): Result dict contains the image data to branch.

        Returns:
            dict: The result dict contains the original image data and copied image data.
        """
        for key1, key2 in self.key_map.items():
            if key1 in results:
                results[key2] = results[key1]
            if key1 in results["img_fields"]:
                results["img_fields"].append(key2)
        return results

    def __repr__(self):
        """Repr."""

        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class TwoCropTransform:
    """TwoCropTransform to combine two pipelines.

    Through `TwoCropTransformHook`, how frequently both pipelines (view0 + view1) is applied can be set.

    Args:
        view0 (list): Pipeline for online network.
        view1 (list): Pipeline for target network.
    """

    def __init__(self, view0: List, view1: List):
        self.view0 = Compose([build_from_cfg(p, PIPELINES) for p in view0])
        self.view1 = Compose([build_from_cfg(p, PIPELINES) for p in view1])
        self.is_both = True

    def __call__(self, results: Dict[str, Any]):
        """Callback function of TwoCropTransform.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes stuffs for training.
                  They have different shape or attribute depending on `self.is_both`.
        """
        if self.is_both:
            results1 = self.view0(deepcopy(results))
            results2 = self.view1(deepcopy(results))

            results = deepcopy(results1)
            results["img"] = np.stack((results1["img"], results2["img"]), axis=0)
            results["gt_semantic_seg"] = np.stack((results1["gt_semantic_seg"], results2["gt_semantic_seg"]), axis=0)
            results["flip"] = [results1["flip"], results2["flip"]]

        else:
            results = self.view0(results)

        results["is_both"] = self.is_both

        return results


@PIPELINES.register_module
class RandomResizedCrop(T.RandomResizedCrop):
    """Wrapper for RandomResizedCrop in torchvision.transforms.

    Since this transformation is applied to PIL Image,
    `NDArrayToPILImage` must be applied first before this is applied.
    """

    def __call__(self, results: Dict[str, Any]):
        """Callback function of RandomResizedCrop.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes transformed img and related information.
        """
        img = results["img"]
        i, j, height, width = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, height, width, self.size, self.interpolation)
        results["img"] = img
        results["img_shape"] = img.size
        for key in results.get("seg_fields", []):
            results[key] = np.array(
                F.resized_crop(
                    Image.fromarray(results[key]),
                    i,
                    j,
                    height,
                    width,
                    self.size,
                    self.interpolation,
                )
            )

        # change order because of difference between numpy and PIL
        w_scale = results["img_shape"][0] / results["ori_shape"][1]
        h_scale = results["img_shape"][1] / results["ori_shape"][0]
        results["scale_factor"] = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        return results


@PIPELINES.register_module
class RandomColorJitter(T.ColorJitter):
    """Wrapper for ColorJitter in torchvision.transforms.

    Since this transformation is applied to PIL Image,
    `NDArrayToPILImage` must be applied first before this is applied.

    Args:
        p (float): Probability for transformation. Defaults to 0.8.
    """

    def __init__(self, p: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, results: Dict[str, Any]):
        """Callback function of ColorJitter.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes transformed img.
        """
        if np.random.random() < self.p:
            results["img"] = self.forward(results["img"])
        return results


@PIPELINES.register_module
class RandomGrayscale(T.RandomGrayscale):
    """Wrapper for RandomGrayscale in torchvision.transforms.

    Since this transformation is applied to PIL Image,
    `NDArrayToPILImage` must be applied first before this is applied.
    """

    def __call__(self, results: Dict[str, Any]):
        """Callback function of RandomGrayscale.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes transformed img.
        """
        results["img"] = self.forward(results["img"])
        return results


@PIPELINES.register_module
class RandomGaussianBlur(T.GaussianBlur):
    """Random Gaussian Blur augmentation inherited from torchvision.transforms.GaussianBlur.

    Since this transformation is applied to PIL Image,
    `NDArrayToPILImage` must be applied first before this is applied.

    Args:
        p (float): Probability for transformation. Defaults to 0.1.
    """

    def __init__(self, p: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, results: Dict[str, Any]):
        """Callback function of GaussianBlur.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes transformed img.
        """
        if np.random.random() < self.p:
            results["img"] = self.forward(results["img"])
        return results


@PIPELINES.register_module
class RandomSolarization:
    """Random Solarization augmentation.

    Args:
        threshold (int): Threshold for solarization. Defaults to 128.
        p (float): Probability for transformation. Defaults to 0.2.
    """

    def __init__(self, threshold: int = 128, p: float = 0.2):
        assert 0 <= p <= 1
        self.threshold = threshold
        self.p = p

    def __call__(self, results: Dict[str, Any]):
        """Callback function of Solarization.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes transformed img.
        """
        if np.random.random() < self.p:
            img = results["img"]
            img = np.where(img < self.threshold, img, 255 - img)
            results["img"] = img
        return results

    def __repr__(self):
        """Set repr of Solarization."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class NDArrayToPILImage:
    """Convert image from numpy to PIL.

    Args:
        keys (list): Keys to be transformed.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, results: Dict[str, Any]):
        """Callback function of NDArrayToPILImage.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes transformed img.
        """
        for key in self.keys:
            img = results[key]
            img = Image.fromarray(img)
            results[key] = img
        return results

    def __repr__(self):
        """Set repr of NDArrayToPILImage."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class PILImageToNDArray:
    """Convert image from PIL to numpy.

    Args:
        keys (list): Keys to be transformed.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, results: Dict[str, Any]):
        """Callback function of PILImageToNDArray.

        Args:
            results (dict): Inputs to be transformed.

        Returns:
            dict: Dictionary that includes transformed img.
        """
        for key in self.keys:
            img = results[key]
            img = np.asarray(img)
            results[key] = img
        return results

    def __repr__(self):
        """Set repr of PILImageToNDArray."""
        repr_str = self.__class__.__name__
        return repr_str
