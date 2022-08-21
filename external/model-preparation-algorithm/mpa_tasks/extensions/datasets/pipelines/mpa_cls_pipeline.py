# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.datasets import PIPELINES
import copy
import numpy as np
from typing import Dict, Any, Optional
from ote_sdk.utils.argument_checks import check_input_parameters_type


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
        dataset_item = results['dataset_item']
        img = dataset_item.numpy
        shape = img.shape

        assert img.shape[0] == results['height'], f"{img.shape[0]} != {results['height']}"
        assert img.shape[1] == results['width'], f"{img.shape[1]} != {results['width']}"

        filename = f"Dataset item index {results['index']}"
        results['filename'] = filename
        results['ori_filename'] = filename
        results['img'] = img
        results['img_shape'] = shape
        results['ori_shape'] = shape
        # Set initial values for default meta_keys
        results['pad_shape'] = shape
        num_channels = 1 if len(shape) < 3 else shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']

        if self.to_float32:
            results['img'] = results['img'].astype(np.float32)

        return results
