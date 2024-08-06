# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
import torch
from datumaro.components.annotation import Ellipse, Image, Mask, Polygon
from torchvision import tv_tensors

from otx.core.data.dataset.base import Transforms
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.types.image import ImageColorChannel
from otx.core.types.label import SegLabelInfo

from .base import OTXDataset

if TYPE_CHECKING:
    from datumaro import Dataset as DmDataset
    from datumaro import DatasetItem


# NOTE: It is copied from https://github.com/openvinotoolkit/datumaro/pull/1409
# It will be replaced in the future.
def _make_index_mask(
    binary_mask: np.ndarray,
    index: int,
    ignore_index: int = 0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Create an index mask from a binary mask by filling a given index value.

    Args:
        binary_mask: Binary mask to create an index mask.
        index: Scalar value to fill the ones in the binary mask.
        ignore_index: Scalar value to fill in the zeros in the binary mask.
            Defaults to 0.
        dtype: Data type for the resulting mask. If not specified,
            it will be inferred from the provided index. Defaults to None.

    Returns:
        np.ndarray: Index mask created from the binary mask.

    Raises:
        ValueError: If dtype is not specified and incompatible scalar types are used for index
            and ignore_index.

    Examples:
        >>> binary_mask = np.eye(2, dtype=np.bool_)
        >>> index_mask = make_index_mask(binary_mask, index=10, ignore_index=255, dtype=np.uint8)
        >>> print(index_mask)
        array([[ 10, 255],
               [255,  10]], dtype=uint8)
    """
    if dtype is None:
        dtype = np.min_scalar_type(index)
        if dtype != np.min_scalar_type(ignore_index):
            raise ValueError

    flipped_zero_np_scalar = ~np.full((), fill_value=0, dtype=dtype)

    # NOTE: This dispatching rule is required for a performance boost
    if ignore_index == flipped_zero_np_scalar:
        flipped_index = ~np.full((), fill_value=index, dtype=dtype)
        return ~(binary_mask * flipped_index)

    mask = binary_mask * np.full((), fill_value=index, dtype=dtype)

    if ignore_index == 0:
        return mask

    return np.where(binary_mask, mask, ignore_index)


def _extract_class_mask(item: DatasetItem, img_shape: tuple[int, int], ignore_index: int) -> np.ndarray:
    """Extract class mask from Datumaro masks.

    This is a temporary workaround and will be replaced with the native Datumaro interfaces
    after some works, e.g., https://github.com/openvinotoolkit/datumaro/pull/1409 are done.

    Args:
        item: Datumaro dataset item having mask annotations.
        img_shape: Image shape (H, W).
        ignore_index: Scalar value to fill in the zeros in the binary mask.

    Returns:
        2D numpy array
    """
    if ignore_index > 255:
        msg = "It is not currently support an ignore index which is more than 255."
        raise ValueError(msg, ignore_index)

    # fill mask with background label if we have Polygon/Ellipse annotations
    fill_value = 0 if isinstance(item.annotations[0], (Ellipse, Polygon)) else ignore_index
    class_mask = np.full(shape=img_shape[:2], fill_value=fill_value, dtype=np.uint8)

    for mask in sorted(
        [ann for ann in item.annotations if isinstance(ann, (Mask, Ellipse, Polygon))],
        key=lambda ann: ann.z_order,
    ):
        index = mask.label

        if index is None:
            msg = "Mask's label index should not be None."
            raise ValueError(msg)

        if isinstance(mask, (Ellipse, Polygon)):
            polygons = np.asarray(mask.as_polygon(), dtype=np.int32).reshape((-1, 1, 2))
            class_index = index + 1  # NOTE: disregard the background index. Objects start from index=1
            this_class_mask = cv2.drawContours(
                class_mask,
                [polygons],
                0,
                (class_index, class_index, class_index),
                thickness=cv2.FILLED,
            )

        elif isinstance(mask, Mask):
            binary_mask = mask.image

            if index is None:
                msg = "Mask's label index should not be None."
                raise ValueError(msg)

            if index > 255:
                msg = "Mask's label index should not be more than 255."
                raise ValueError(msg, index)

            this_class_mask = _make_index_mask(
                binary_mask=binary_mask,
                index=index,
                ignore_index=ignore_index,
                dtype=np.uint8,
            )

            if this_class_mask.shape != img_shape:
                this_class_mask = cv2.resize(
                    this_class_mask,
                    dsize=(img_shape[1], img_shape[0]),  # NOTE: cv2.resize() uses (width, height) format
                    interpolation=cv2.INTER_NEAREST,
                )

        class_mask = np.where(this_class_mask != ignore_index, this_class_mask, class_mask)

    return class_mask


class OTXSegmentationDataset(OTXDataset[SegDataEntity]):
    """OTXDataset class for segmentation task."""

    def __init__(
        self,
        dm_subset: DmDataset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        ignore_index: int = 255,
    ) -> None:
        super().__init__(
            dm_subset,
            transforms,
            mem_cache_handler,
            mem_cache_img_max_size,
            max_refetch,
            image_color_channel,
            stack_images,
            to_tv_image,
        )

        if self.has_polygons and "background" not in [label_name.lower() for label_name in self.label_info.label_names]:
            # insert background class at index 0 since polygons represent only objects
            self.label_info.label_names.insert(0, "background")

        self.label_info = SegLabelInfo(
            label_names=self.label_info.label_names,
            label_groups=self.label_info.label_groups,
            ignore_index=ignore_index,
        )
        self.ignore_index = ignore_index

    @property
    def has_polygons(self) -> bool:
        """Check if the dataset has polygons in annotations."""
        ann_types = {str(ann_type).split(".")[-1] for ann_type in self.dm_subset.ann_types()}
        if ann_types & {"polygon", "ellipse"}:
            return True
        return False

    def _get_item_impl(self, index: int) -> SegDataEntity | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []
        img_data, img_shape = self._get_img_data_and_shape(img)
        if item.annotations:
            extracted_mask = _extract_class_mask(item=item, img_shape=img_shape, ignore_index=self.ignore_index)
            masks = tv_tensors.Mask(extracted_mask[None])
        else:
            # semi-supervised learning, unlabeled dataset
            masks = torch.tensor([[0]])

        entity = SegDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            masks=masks,
        )
        transformed_entity = self._apply_transforms(entity)
        return transformed_entity.wrap(masks=transformed_entity.masks[0]) if transformed_entity else None

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect SegDataEntity into SegBatchDataEntity in data loader."""
        return partial(SegBatchDataEntity.collate_fn, stack_images=self.stack_images)
