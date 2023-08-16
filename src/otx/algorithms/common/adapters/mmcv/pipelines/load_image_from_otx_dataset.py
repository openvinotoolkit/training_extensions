"""Pipeline element that loads an image from a OTX Dataset on the fly."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple

import numpy as np

from otx.algorithms.common.utils.data import get_image
from otx.core.data.caching import MemCacheHandlerError, MemCacheHandlerSingleton

_CACHE_DIR = TemporaryDirectory(prefix="img-cache-")  # pylint: disable=consider-using-with

# TODO: refactoring to common modules


class LoadImageFromOTXDataset:
    """Pipeline element that loads an image from a OTX Dataset on the fly.

    Can do conversion to float 32 if needed.
    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the image
        results['dataset_id']: id of the dataset to which the item belongs
        results['index']: index of the item in the dataset

    Args:
        to_float32 (bool, optional): True to convert images to fp32. defaults to False.
        enable_memcache (bool, optional): True to enable in-memory cache. defaults to False.
    """

    def __init__(self, to_float32: bool = False, enable_memcache: bool = False):
        self._to_float32 = to_float32
        self._enable_memcache = enable_memcache
        try:
            self._mem_cache_handler = MemCacheHandlerSingleton.get()
        except MemCacheHandlerError:
            # Create a null handler
            MemCacheHandlerSingleton.create(mode="null", mem_size=0)
            self._mem_cache_handler = MemCacheHandlerSingleton.get()

    @staticmethod
    def _get_unique_key(results: Dict[str, Any]) -> Tuple:
        """Returns unique key of data item based on the contents."""
        # TODO: We should improve it by assigning an unique id to DatasetItemEntity.
        # This is because there is a case which
        # d_item.media.path is None, but d_item.media.data is not None
        d_item = results["dataset_item"]
        return d_item.media.path, d_item.roi.id

    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadImageFromOTXDataset."""
        img = None
        if self._enable_memcache:
            key = self._get_unique_key(results)
            img, meta = self._mem_cache_handler.get(key)

        if img is None:
            # Get image (possibly from file cache)
            img = get_image(results, _CACHE_DIR.name, to_float32=False)
            if self._enable_memcache:
                self._mem_cache_handler.put(key, img)

        if self._to_float32:
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
        results["entity_id"] = results.get("entity_id")
        results["label_id"] = results.get("label_id")

        return results


class LoadResizeDataFromOTXDataset(LoadImageFromOTXDataset):
    """Load and resize image & annotation with cache support.

    This base operation loads image and optionally loads annotations.
    Then, resize the image and annotation accordingly if resize_cfg given & it's beneficial,
    e.g. the size is smaller than original input size.
    Finally, if enabled, cache the result and use pre-computed ones from next iterations.

    Args:
        load_img_cfg (Dict): Creates image loading operation based on the config
        load_ann_cfg (Dict, optional): Optionally creates annotation loading operation based on the config.
            Defaults to None.
        resize_cfg (Dict, optional): Optionally creates resize operation based on the config. Defaults to None.
        enable_memcache (bool, optional): True to enable in-memory cache. Defaults to False.
    """

    def __init__(
        self,
        load_img_cfg: Dict,
        load_ann_cfg: Optional[Dict] = None,
        resize_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._load_img_cfg = load_img_cfg.copy()
        self._load_img_cfg["enable_memcache"] = False  # will use outer cache
        self._load_img_op = self._create_load_img_op(self._load_img_cfg)
        self._load_ann_op = self._create_load_ann_op(load_ann_cfg)
        self._downscale_only = resize_cfg.pop("downscale_only", False) if resize_cfg else False
        self._resize_op = self._create_resize_op(resize_cfg)
        if self._resize_op is not None:
            self._resize_shape = resize_cfg.get("size", resize_cfg.get("img_scale"))
            assert isinstance(self._resize_shape, tuple), f"Random scale is not supported by {self.__class__.__name__}"
        else:
            self._resize_shape = None

    def _create_load_img_op(self, cfg: Dict) -> Any:
        """Creates image loading operation."""
        return LoadImageFromOTXDataset(**cfg)  # Should be overrided in task-specific implementation if needed

    def _create_load_ann_op(self, cfg: Optional[Dict]) -> Optional[Any]:
        """Creates annotation loading operation."""
        return None  # Should be overrided in task-specific implementation

    def _create_resize_op(self, cfg: Optional[Dict]) -> Optional[Any]:
        """Creates resize operation."""
        return None  # Should be overrided in task-specific implementation

    def _load_img(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load image and fill the results dict."""
        return self._load_img_op(results)

    def _load_ann_if_any(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load annotations and fill the results dict."""
        if self._load_ann_op is None:
            return results
        return self._load_ann_op(results)

    def _resize_img_ann_if_any(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Resize image and annotations if needed and fill the results dict."""
        if self._resize_op is None:
            return results
        original_shape = results.get("img_shape", self._resize_shape)
        if original_shape is None:
            return results
        if self._downscale_only:
            if original_shape[0] * original_shape[1] <= self._resize_shape[0] * self._resize_shape[1]:
                # No benfit of early resizing if resize_shape is larger than original_shape
                return results
        return self._resize_op(results)

    def _load_cache(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to load pre-computed results from cache."""
        if not self._enable_memcache:
            return None
        key = self._get_unique_key(results)
        img, meta = self._mem_cache_handler.get(key)
        if img is None or meta is None:
            return None
        results = meta.copy()
        results["img"] = img
        return results

    def _save_cache(self, results: Dict[str, Any]):
        """Try to save pre-computed results to cache."""
        if not self._enable_memcache:
            return
        key = self._get_unique_key(results)
        meta = results.copy()
        img = meta.pop("img")
        self._mem_cache_handler.put(key, img, meta)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Callback function."""
        cached_results = self._load_cache(results)
        if cached_results:
            return cached_results
        results = self._load_img(results)
        results = self._load_ann_if_any(results)
        results = self._resize_img_ann_if_any(results)
        self._save_cache(results)
        return results
