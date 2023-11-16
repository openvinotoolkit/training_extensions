"""Pipeline element that loads an image from a OTX Dataset on the fly."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mmcv.transforms import BaseTransform

from otx.v2.adapters.datumaro.caching.mem_cache_handler import (
    MemCacheHandlerBase,
    MemCacheHandlerError,
    MemCacheHandlerSingleton,
)
from otx.v2.api.entities.utils.data_utils import get_image

_CACHE_DIR = TemporaryDirectory(prefix="img-cache-")


class LoadImageFromOTXDataset:
    """Pipeline element that loads an image from a OTX Dataset on the fly.

    Can do conversion to float 32 if needed.
    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the image
        results['dataset_id']: id of the dataset to which the item belongs
        results['index']: index of the item in the dataset

    :param to_float32: optional bool, True to convert images to fp32. defaults to False
    """

    def __init__(self, to_float32: bool = False, enable_memcache: bool = True) -> None:
        self.to_float32 = to_float32
        self._enable_memcache = enable_memcache

    @staticmethod
    def _get_unique_key(results: dict) -> tuple:
        # This is because there is a case which
        # d_item.media.path is None, but d_item.media.data is not None
        if "cache_key" in results:
            return results["cache_key"]
        d_item = results["dataset_item"]
        results["cache_key"] = d_item.media.path, d_item.id
        return results["cache_key"]

    def _get_memcache_handler(self) -> MemCacheHandlerBase:
        """Get memcache handler."""
        try:
            mem_cache_handler = MemCacheHandlerSingleton.get()
        except MemCacheHandlerError:
            # Create a null handler
            MemCacheHandlerSingleton.create(mode="null", mem_size=0)
            mem_cache_handler = MemCacheHandlerSingleton.get()

        return mem_cache_handler

    def __call__(self, results: dict) -> dict:
        """Callback function of LoadImageFromOTXDataset."""
        img = None
        mem_cache_handler = self._get_memcache_handler()

        if self._enable_memcache:
            key = self._get_unique_key(results)
            img, meta = mem_cache_handler.get(key)

        if img is None:
            # Get image (possibly from cache)
            img = get_image(results, _CACHE_DIR.name, to_float32=False)
            if self._enable_memcache:
                mem_cache_handler.put(key, img)

        if self.to_float32:
            img = img.astype(np.float32)
        shape = img.shape

        filename = f"Dataset item index {results['index']}"
        results["filename"] = filename
        results["ori_filename"] = filename
        results["img"] = img
        results["img_shape"] = shape[:2]
        results["ori_shape"] = shape[:2]
        results["height"] = shape[0]
        results["width"] = shape[1]
        # Set initial values for default meta_keys
        results["pad_shape"] = shape[:2]
        num_channels = 1 if len(shape) < 3 else shape[2]
        results["img_norm_cfg"] = {
            "mean": np.zeros(num_channels, dtype=np.float32),
            "std": np.ones(num_channels, dtype=np.float32),
            "to_rgb": False,
        }
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
        load_ann_cfg (Dict, optional): Optionally creates annotation loading operation based on the config.
            Defaults to None.
        resize_cfg (Dict, optional): Optionally creates resize operation based on the config. Defaults to None.
    """

    def __init__(
        self,
        load_ann_cfg: dict | None = None,
        resize_cfg: dict | None = None,
        eval_mode: bool = False,
        **kwargs,
    ):
        self.eval_mode = eval_mode
        self._enable_outer_memcache = kwargs.get("enable_memcache", True)
        kwargs["enable_memcache"] = False  # will use outer cache
        super().__init__(**kwargs)
        self._load_ann_op = self._create_load_ann_op(load_ann_cfg)
        self._downscale_only = resize_cfg.pop("downscale_only", False) if resize_cfg else False
        self._resize_op = self._create_resize_op(resize_cfg)
        if self._resize_op is not None and resize_cfg is not None:
            self._resize_shape = resize_cfg.get("scale", None)
            if isinstance(self._resize_shape, int):
                self._resize_shape = (self._resize_shape, self._resize_shape)
            if isinstance(self._resize_shape, list):
                self._resize_shape = tuple(self._resize_shape)
            assert isinstance(self._resize_shape, tuple), f"Random scale is not supported by {self.__class__.__name__}"
            self._resize_op.scale = self._resize_shape
        else:
            self._resize_shape = None

    def _create_load_ann_op(self, cfg: dict | None = None) -> BaseTransform | None:
        """Creates annotation loading operation."""
        raise NotImplementedError

    def _create_resize_op(self, cfg: dict | None = None) -> BaseTransform | None:
        """Creates resize operation."""
        raise NotImplementedError

    def _load_img(self, results: dict) -> dict:
        """Load image and fill the results dict."""
        return super().__call__(results)  # Use image load logic from base class

    def _load_ann_if_any(self, results: dict) -> dict:
        """Load annotations and fill the results dict."""
        if self._load_ann_op is None:
            return results
        return self._load_ann_op(results)

    def _resize_img_ann_if_any(self, results: dict) -> dict:
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

    def _load_cache(self, results: dict) -> dict | None:
        """Try to load pre-computed results from cache."""

        if not self._enable_outer_memcache:
            return None
        key = self._get_unique_key(results)

        mem_cache_handler = self._get_memcache_handler()
        img, meta = mem_cache_handler.get(key)
        if img is None or meta is None:
            return None
        dataset_item = results.pop("dataset_item")
        results = meta.copy()
        results["img"] = img
        results["dataset_item"] = dataset_item
        return results

    def _save_cache(self, results: dict) -> None:
        """Try to save pre-computed results to cache."""
        if not self._enable_outer_memcache:
            return

        key = self._get_unique_key(results)
        meta = results.copy()
        img = meta.pop("img")

        mem_cache_handler = self._get_memcache_handler()
        mem_cache_handler.put(key, img, meta)

    def __call__(self, results: dict) -> dict:
        """Callback function."""
        _results = results.copy()
        cached_results = self._load_cache(_results)
        if cached_results:
            return cached_results
        if self.eval_mode:
            _results = self._load_img(_results)
            _results = self._resize_img_ann_if_any(_results)
            _results = self._load_ann_if_any(_results)
        else:
            _results = self._load_img(_results)
            _results = self._load_ann_if_any(_results)
            _results = self._resize_img_ann_if_any(_results)

        # Common post-processing steps
        if _results is not None:
            _results.pop("dataset_item", None)
        if isinstance(_results, dict):
            self._save_cache(_results)
        return _results
