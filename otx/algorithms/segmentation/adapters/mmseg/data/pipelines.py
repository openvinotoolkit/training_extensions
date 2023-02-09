"""Collection Pipeline for segmentation task."""
# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
import tempfile
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from mmcv.utils import build_from_cfg
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import Compose
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

from otx.algorithms.common.utils.data import get_image
from otx.api.utils.argument_checks import check_input_parameters_type

from .dataset import get_annotation_mmseg_format

_CACHE_DIR = tempfile.TemporaryDirectory(prefix="img-cache-")  # pylint: disable=consider-using-with


@PIPELINES.register_module()
class LoadImageFromOTXDataset:
    """Pipeline element that loads an image from a OTX Dataset on the fly. Can do conversion to float 32 if needed.

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
        """Callback function LoadImageFromOTXDataset."""
        # Get image (possibly from cache)
        img = get_image(results, _CACHE_DIR.name, to_float32=self.to_float32)
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
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        results["img_fields"] = ["img"]

        return results


@PIPELINES.register_module()
class LoadAnnotationFromOTXDataset:
    """Pipeline element that loads an annotation from a OTX Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project

    """

    def __init__(self):
        pass

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results["dataset_item"]
        labels = results["ann_info"]["labels"]

        ann_info = get_annotation_mmseg_format(dataset_item, labels)

        results["gt_semantic_seg"] = ann_info["gt_semantic_seg"]
        results["seg_fields"].append("gt_semantic_seg")

        return results


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

    @check_input_parameters_type()
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
                F.resized_crop(Image.fromarray(results[key]), i, j, height, width, self.size, self.interpolation)
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
