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

from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from mmcv.utils import build_from_cfg
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import Compose, to_tensor
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

from otx.api.utils.argument_checks import check_input_parameters_type

from .dataset import get_annotation_mmseg_format


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
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        results["img_fields"] = ["img"]

        if self.to_float32:
            results["img"] = results["img"].astype(np.float32)

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

    :param view0: Pipeline for online network.
    :param view1: Pipeline for target network.
    """

    def __init__(self, view0: List, view1: List):
        self.pipeline1 = Compose([build_from_cfg(p, PIPELINES) for p in view0])
        self.pipeline2 = Compose([build_from_cfg(p, PIPELINES) for p in view1])

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of TwoCropTransform.

        :param results: Inputs to be transformed.
        """
        results1 = self.pipeline1(deepcopy(results))
        results2 = self.pipeline2(deepcopy(results))

        results = deepcopy(results1)
        results["img"] = to_tensor(
            np.ascontiguousarray(np.stack((results1["img"], results2["img"]), axis=0).transpose(0, 3, 1, 2))
        )
        results["gt_semantic_seg"] = to_tensor(
            np.ascontiguousarray(
                np.stack((results1["gt_semantic_seg"], results2["gt_semantic_seg"]), axis=0).transpose(0, 1, 2)
            )
        )
        results["flip"] = [results1["flip"], results2["flip"]]

        return results


@PIPELINES.register_module
class RandomResizedCrop(T.RandomResizedCrop):
    """Wrapper for RandomResizedCrop in torchvision.transforms."""

    def __call__(self, results: Dict[str, Any]):
        """Callback function of RandomResizedCrop.

        :param results: Inputs to be transformed.
        """
        img = Image.fromarray(results["img"])

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = np.array(F.resized_crop(img, i, j, h, w, self.size, self.interpolation))
        results["img"] = img
        results["img_shape"] = img.shape
        for key in results.get("seg_fields", []):
            results[key] = np.array(
                F.resized_crop(Image.fromarray(results[key]), i, j, h, w, self.size, self.interpolation)
            )

        w_scale = results["img_shape"][1] / results["ori_shape"][1]
        h_scale = results["img_shape"][0] / results["ori_shape"][0]
        results["scale_factor"] = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        return results


@PIPELINES.register_module
class ColorJitter(T.ColorJitter):
    """Wrapper for ColorJitter in torchvision.transforms."""

    def __call__(self, results: Dict[str, Any]):
        """Callback function of ColorJitter.

        :param results: Inputs to be transformed.
        """
        results["img"] = np.array(self.forward(Image.fromarray(results["img"])))
        return results


@PIPELINES.register_module
class RandomGrayscale(T.RandomGrayscale):
    """Wrapper for RandomGrayscale in torchvision.transforms."""

    def __call__(self, results: Dict[str, Any]):
        """Callback function of RandomGrayscale.

        :param results: Inputs to be transformed.
        """
        results["img"] = np.array(self.forward(Image.fromarray(results["img"])))
        return results


@PIPELINES.register_module
class GaussianBlur(T.GaussianBlur):
    """Wrapper for GaussianBlur in torchvision.transforms."""

    def __call__(self, results: Dict[str, Any]):
        """Callback function of GaussianBlur.

        :param results: Inputs to be transformed.
        """
        results["img"] = np.array(self.forward(Image.fromarray(results["img"])))

        return results


@PIPELINES.register_module
class Solarization:
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733.

    :param threshold: Threshold for solarization, defaults to 128
    """

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, results: Dict[str, Any]):
        """Callback function of Solarization.

        :param results: inputs to be transformed.
        """
        img = results["img"]
        img = np.where(img < self.threshold, img, 255 - img)
        results["img"] = img
        return results

    def __repr__(self):
        """Set repr of Solarization."""
        repr_str = self.__class__.__name__
        return repr_str
