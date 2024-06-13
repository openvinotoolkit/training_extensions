# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for OTXDataset."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Generic, Iterator, List, Union

import cv2
import numpy as np
from datumaro.components.annotation import AnnotationType
from datumaro.components.media import ImageFromFile
from datumaro.util.image import IMAGE_BACKEND, IMAGE_COLOR_CHANNEL, ImageBackend
from datumaro.util.image import ImageColorChannel as DatumaroImageColorChannel
from torch.utils.data import Dataset

from otx.core.data.entity.base import T_OTXDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER
from otx.core.data.transform_libs.torchvision import Compose
from otx.core.types.image import ImageColorChannel
from otx.core.types.label import LabelInfo, NullLabelInfo

if TYPE_CHECKING:
    from datumaro import DatasetSubset, Image

    from otx.core.data.mem_cache import MemCacheHandlerBase

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


@contextmanager
def image_decode_context() -> Iterator[None]:
    """Change Datumaro image decode context.

    Use PIL Image decode because of performance issues.
    With this context, `dm.Image.data` will return BGR numpy image tensor.
    """
    ori_image_backend = IMAGE_BACKEND.get()
    ori_image_color_scale = IMAGE_COLOR_CHANNEL.get()

    IMAGE_BACKEND.set(ImageBackend.PIL)
    # TODO(vinnamki): This should be changed to
    # if to_rgb:
    #     IMAGE_COLOR_CHANNEL.set(DatumaroImageColorChannel.COLOR_RGB)
    # else:
    #     IMAGE_COLOR_CHANNEL.set(DatumaroImageColorChannel.COLOR_BGR)
    # after merging https://github.com/openvinotoolkit/datumaro/pull/1501
    IMAGE_COLOR_CHANNEL.set(DatumaroImageColorChannel.COLOR_RGB)

    yield

    IMAGE_BACKEND.set(ori_image_backend)
    IMAGE_COLOR_CHANNEL.set(ori_image_color_scale)


class OTXDataset(Dataset, Generic[T_OTXDataEntity]):
    """Base OTXDataset.

    Defines basic logic for OTX datasets.

    Args:
        dm_subset: Datumaro subset of a dataset
        transforms: Transforms to apply on images
        mem_cache_handler: Handler of the images cache
        mem_cache_img_max_size: Max size of images to put in cache
        max_refetch: Maximum number of images to fetch in cache
        image_color_channel: Color channel of images
        stack_images: Whether or not to stack images in collate function in OTXBatchData entity.

    """

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
    ) -> None:
        self.dm_subset = dm_subset
        self.transforms = transforms
        self.mem_cache_handler = mem_cache_handler
        self.mem_cache_img_max_size = mem_cache_img_max_size
        self.max_refetch = max_refetch
        self.image_color_channel = image_color_channel
        self.stack_images = stack_images
        self.to_tv_image = to_tv_image
        if self.dm_subset.categories():
            self.label_info = LabelInfo.from_dm_label_groups(self.dm_subset.categories()[AnnotationType.label])
        else:
            self.label_info = NullLabelInfo()

    def __len__(self) -> int:
        return len(self.dm_subset)

    def _sample_another_idx(self) -> int:
        return np.random.default_rng().integers(0, len(self))

    def _apply_transforms(self, entity: T_OTXDataEntity) -> T_OTXDataEntity | None:
        if isinstance(self.transforms, Compose):
            if self.to_tv_image:
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

        with image_decode_context():
            img_data = (
                img.data
                if self.image_color_channel == ImageColorChannel.RGB
                else cv2.cvtColor(img.data, cv2.COLOR_RGB2BGR)
            )

        if img_data is None:
            msg = "Cannot get image data"
            raise RuntimeError(msg)

        img_data = self._cache_img(key=key, img_data=img_data.astype(np.uint8))

        return img_data, img_data.shape[:2]

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
