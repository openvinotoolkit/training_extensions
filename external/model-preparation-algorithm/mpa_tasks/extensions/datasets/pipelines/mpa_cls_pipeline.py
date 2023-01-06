# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict

import numpy as np
from mmcls.datasets import PIPELINES
from mpa_tasks.utils.data_utils import clean_up_cache_dir, get_image
from ote_sdk.utils.argument_checks import check_input_parameters_type

_CACHE_DIR = "/tmp/cls-img-cache"
clean_up_cache_dir(_CACHE_DIR)  # Clean up cache directory per process launch


# Temporary copy from detection_tasks
# TODO: refactoring to common modules
@PIPELINES.register_module()
class LoadImageFromOTEDataset:
    """
    Pipeline element that loads an image from a OTE Dataset on the fly. Can do conversion to float 32 if needed.

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
        # Get image (possibly from cache)
        img = get_image(results, _CACHE_DIR, to_float32=self.to_float32)
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

        return results
