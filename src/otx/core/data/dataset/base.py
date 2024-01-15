# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for OTXDataset."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Generic, List, Union

import cv2
import numpy as np
from datumaro.components.annotation import AnnotationType
from datumaro.components.media import ImageFromFile
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose

from otx.core.data.entity.base import T_OTXDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER
from otx.core.types.image import ImageColorChannel

if TYPE_CHECKING:
    from datumaro import DatasetSubset, Image

    from otx.core.data.mem_cache import MemCacheHandlerBase

Transforms = Union[Compose, Callable, List[Callable]]


@dataclass
class LabelInfo:
    """Object to represent label information."""

    label_names: list[str]

    @property
    def num_classes(self) -> int:
        """Return number of labels."""
        return len(self.label_names)

    @classmethod
    def from_num_classes(cls, num_classes: int) -> LabelInfo:
        """Create this object from the number of classes.

        Args:
            num_classes: Number of classes

        Returns:
            LabelInfo(label_names=["label_0", ...])
        """
        return LabelInfo(label_names=[f"label_{idx}" for idx in range(num_classes)])


class OTXDataset(Dataset, Generic[T_OTXDataEntity]):
    """Base OTXDataset."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
    ) -> None:
        self.dm_subset = dm_subset
        self.ids = [item.id for item in dm_subset]
        self.transforms = transforms
        self.mem_cache_handler = mem_cache_handler
        self.mem_cache_img_max_size = mem_cache_img_max_size
        self.max_refetch = max_refetch
        self.image_color_channel = image_color_channel

        self.meta_info = LabelInfo(
            label_names=[category.name for category in self.dm_subset.categories()[AnnotationType.label]],
        )

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
        key = img.path if isinstance(img, ImageFromFile) else id(img)

        if (img_data := self.mem_cache_handler.get(key=key)[0]) is not None:
            return img_data, img_data.shape[:2]

        # TODO(vinnamkim): This is a temporal approach
        # There is an upcoming Datumaro patch here for this
        # https://github.com/openvinotoolkit/datumaro/pull/1194
        img_data = (
            self._read_from_bytes(img_bytes) if (img_bytes := img.bytes) is not None else self._read_from_data(img.data)
        )

        if img_data is None:
            msg = "Cannot get image data"
            raise RuntimeError(msg)

        img_data = self._cache_img(key=key, img_data=img_data)

        return img_data, img_data.shape[:2]

    def _read_from_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Read an image from `img.bytes`."""
        img_data = np.asarray(PILImage.open(BytesIO(img_bytes)).convert("RGB"))

        if self.image_color_channel == ImageColorChannel.BGR:
            return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        return img_data

    def _read_from_data(self, img_data: np.ndarray | None) -> np.ndarray | None:
        """This function is required for `img.data` (not read by PIL)."""
        if img_data is None:
            return None

        # TODO(vinnamki): dm.ImageFromData forces `img_data` to have `np.float32` type. #noqa: TD003
        # This behavior will be removed in the Datumaro side.
        if img_data.dtype == np.float32:
            img_data = img_data.astype(np.uint8)

        if img_data.shape[-1] == 4:
            conversion = cv2.COLOR_BGRA2RGB if self.image_color_channel == ImageColorChannel.RGB else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(img_data, conversion)
        if len(img_data.shape) == 2:
            conversion = cv2.COLOR_GRAY2RGB if self.image_color_channel == ImageColorChannel.RGB else cv2.COLOR_GRAY2BGR
            return cv2.cvtColor(img_data, conversion)
        if self.image_color_channel == ImageColorChannel.RGB:
            return cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        return img_data

    def _cache_img(self, key: str | int, img_data: np.ndarray) -> np.ndarray:
        """Cache an image after resizing.

        If there is available space in the memory pool, the input image is cached.
        Before caching, the input image is resized if it is larger than the maximum image size
        specified by the memory caching handler.
        Otherwise, the input image is directly cached.
        After caching, the processed image data is returned.

        Args:
            key: The key associated with the image.
            img_data: The image data to be cached.

        Returns:
            The resized image if it was resized. Otherwise, the original image.
        """
        if self.mem_cache_handler.frozen:
            return img_data

        if self.mem_cache_img_max_size is None:
            self.mem_cache_handler.put(key=key, data=img_data, meta=None)
            return img_data

        height, width = img_data.shape[:2]
        max_height, max_width = self.mem_cache_img_max_size

        if height <= max_height and width <= max_width:
            self.mem_cache_handler.put(key=key, data=img_data, meta=None)
            return img_data

        # Preserve the image size ratio and fit to max_height or max_width
        # e.g. (1000 / 2000 = 0.5, 1000 / 1000 = 1.0) => 0.5
        # h, w = 2000 * 0.5 => 1000, 1000 * 0.5 => 500, bounded by max_height
        min_scale = min(max_height / height, max_width / width)
        new_height, new_width = int(min_scale * height), int(min_scale * width)
        resized_img = cv2.resize(
            src=img_data,
            dsize=(new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        self.mem_cache_handler.put(
            key=key,
            data=resized_img,
            meta=None,
        )
        return resized_img

    @abstractmethod
    def _get_item_impl(self, idx: int) -> T_OTXDataEntity | None:
        pass

    @property
    @abstractmethod
    def collate_fn(self) -> Callable:
        """Collection function to collect OTXDataEntity into OTXBatchDataEntity in data loader."""
