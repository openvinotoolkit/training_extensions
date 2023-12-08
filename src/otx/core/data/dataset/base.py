# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for OTXDataset."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Generic, List, Union

import cv2
import numpy as np
from datumaro.components.media import ImageFromFile
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose

from otx.core.data.entity.base import T_OTXDataEntity
from otx.core.data.mem_cache import MemCacheHandlerSingleton

if TYPE_CHECKING:
    from datumaro import DatasetSubset, Image

    from otx.core.data.mem_cache import MemCacheHandlerBase

Transforms = Union[Compose, Callable, List[Callable]]


class OTXDataset(Dataset, Generic[T_OTXDataEntity]):
    """Base OTXDataset."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
    ) -> None:
        self.dm_subset = dm_subset
        self.ids = [item.id for item in dm_subset]
        self.transforms = transforms
        self.mem_cache_img_max_size = mem_cache_img_max_size
        self.max_refetch = max_refetch

    def __len__(self) -> int:
        return len(self.ids)

    def _sample_another_idx(self) -> int:
        return np.random.default_rng().integers(0, len(self))

    def _apply_transforms(self, entity: T_OTXDataEntity) -> T_OTXDataEntity | None:
        if isinstance(self.transforms, Compose):
            entity = entity.to_tv_image()
            return self.transforms(entity)
        if isinstance(self.transforms, Iterable):
            return self._iterable_transforms(entity)
        if callable(self.transforms):
            return self.transforms(entity)

        raise TypeError(self.transforms)

    def _iterable_transforms(self, item: T_OTXDataEntity) -> T_OTXDataEntity | None:
        if not isinstance(self.transforms, list):
            raise TypeError(item)

        results = item
        for transform in self.transforms:
            results = transform(results)
            # MMCV transform can produce None. Please see
            # https://github.com/open-mmlab/mmengine/blob/26f22ed283ae4ac3a24b756809e5961efe6f9da8/mmengine/dataset/base_dataset.py#L59-L66
            if results is None:
                return None

        return results

    def __getitem__(self, index: int) -> T_OTXDataEntity:
        for _ in range(self.max_refetch):
            results = self._get_item_impl(index)

            if results is not None:
                return results

            index = self._sample_another_idx()

        msg = f"Reach the maximum refetch number ({self.max_refetch})"
        raise RuntimeError(msg)

    def _get_img_data_and_shape(self, img: Image) -> tuple[np.ndarray, tuple[int, int]]:
        handler = MemCacheHandlerSingleton.get()

        key = img.path if isinstance(img, ImageFromFile) else id(img)

        if handler is not None and (img_data := handler.get(key=key)[0]) is not None:
            return img_data, img_data.shape[:2]

        img_data = img.data

        # TODO(vinnamkim): This is a temporal approach
        # There is an upcoming Datumaro patch here for this
        # https://github.com/openvinotoolkit/datumaro/pull/1194
        if img_data.shape[-1] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        if len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)

        if handler is not None:
            self._cache_img(
                handler=handler,
                key=key,
                img_data=img_data,
                img_size=img.size,
            )

        return img_data, img_data.shape[:2]

    def _cache_img(
        self,
        handler: MemCacheHandlerBase,
        key: str | int,
        img_data: np.ndarray,
        img_size: tuple[int, int],
    ) -> None:
        if self.mem_cache_img_max_size is None:
            handler.put(key=key, data=img_data, meta=None)
            return

        height, width = img_size
        max_height, max_width = self.mem_cache_img_max_size

        if height <= max_height and width <= max_width:
            handler.put(key=key, data=img_data, meta=None)
            return

        # Preserve the image size ratio and fit to max_height or max_width
        # e.g. (1000 / 2000 = 0.5, 1000 / 1000 = 1.0) => 0.5
        # h, w = 2000 * 0.5 => 1000, 1000 * 0.5 => 500, bounded by max_height
        min_scale = min(max_height / height, max_width / width)
        new_height, new_width = int(min_scale * height), int(min_scale * width)

        handler.put(
            key=key,
            data=cv2.resize(
                src=img_data,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            ),
            meta=None,
        )

    @abstractmethod
    def _get_item_impl(self, idx: int) -> T_OTXDataEntity | None:
        pass

    @property
    @abstractmethod
    def collate_fn(self) -> Callable:
        """Collection function to collect OTXDataEntity into OTXBatchDataEntity in data loader."""
