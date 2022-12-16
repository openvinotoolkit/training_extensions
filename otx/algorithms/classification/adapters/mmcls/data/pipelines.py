"""Collection Pipeline for classification task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict

import numpy as np
from mmcls.datasets import PIPELINES

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
class PILImageToNDArray:
    """Pipeline element that converts an PIL image into numpy array.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['img']: PIL type image in data pipeline.

    :param keys: optional list, support multiple image converting. defaults to ["img"].
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
class BranchField:
    """Pipeline element that append given key in results["img_fields"] enable augmentations for mmcls.

    Expected entries in the "results" dict that should be passed to this pipeline element are:
        results["img_fields"]: list of target image fields. ex) ["img", "img_weak"]

    :param keys: optional list, support appending multiple image fields.
    """

    def __init__(self, key_map=None):
        self.key_map = key_map

    def __call__(self, results):
        """Callback function of BranchField."""
        for src, dst in self.key_map.items():
            if src in results["img_fields"]:
                results["img_fields"].append(dst)
        return results

    def __repr__(self):
        """Repr function of BranchField."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class BranchImage(BranchField):
    """Pipeline element that append given key in results to bypass incoming augmentations.

    Expected entries in the "results" dict that should be passed to this pipeline element are:
        results["img"]: target image to copy.

    """

    def __call__(self, results):
        """Callback function of BranchImage."""
        for src, dst in self.key_map.items():
            if src in results:
                results[dst] = results[src]
        return results
