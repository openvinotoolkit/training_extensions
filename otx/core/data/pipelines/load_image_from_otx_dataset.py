"""Pipeline element that loads an image from a OTX Dataset on the fly."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import numpy as np

from otx.algorithms.common.utils.data import get_image
from otx.api.utils.argument_checks import check_input_parameters_type

from ..caching import MemCacheHandlerError, MemCacheHandlerSingleton

_CACHE_DIR = TemporaryDirectory(prefix="img-cache-")  # pylint: disable=consider-using-with

# TODO: refactoring to common modules
# TODO: refactoring to Sphinx style.


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
        try:
            self.mem_cache_handler = MemCacheHandlerSingleton.get()
        except MemCacheHandlerError:
            # Create a null handler
            MemCacheHandlerSingleton.create(mode="null", mem_size=0)
            self.mem_cache_handler = MemCacheHandlerSingleton.get()

    @staticmethod
    def _get_unique_key(results: Dict[str, Any]) -> Tuple:
        # TODO: We should improve it by assigning an unique id to DatasetItemEntity.
        # This is because there is a case which
        # d_item.media.path is None, but d_item.media.data is not None
        d_item = results["dataset_item"]
        return d_item.media.path, d_item.roi.id

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadImageFromOTXDataset."""
        key = self._get_unique_key(results)

        img = self.mem_cache_handler.get(key)

        if img is None:
            # Get image (possibly from cache)
            img = get_image(results, _CACHE_DIR.name, to_float32=False)
            self.mem_cache_handler.put(key, img)

        if self.to_float32:
            img = img.astype(np.float32)
        shape = img.shape

        if img.shape[0] != results["height"]:
            results["height"] = img.shape[0]

        if img.shape[1] != results["width"]:
            results["width"] = img.shape[1]

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
