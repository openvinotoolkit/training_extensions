# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXSegmentationDataset."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from datumaro.components.annotation import Image, Mask
from torchvision import tv_tensors

from otx.core.data.dataset.base import LabelInfo, Transforms
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.types.image import ImageColorChannel

from .base import OTXDataset

if TYPE_CHECKING:
    from datumaro import DatasetSubset


@dataclass
class SegMetaInfo(LabelInfo):
    """Meta information of Semantic Segmentation."""

    def __init__(self, label_names: list[str], label_groups: list[list[str]]) -> None:
        if not any(word.lower() == "background" for word in label_names):
            msg = (
                "Currently, no background label exists for `label_names`. "
                "Segmentation requires a background label. "
                "To do this, `Background` is added at index 0 of `label_names`."
            )
            warnings.warn(msg, stacklevel=2)
            label_names.insert(0, "Background")
        super().__init__(label_names, label_groups)


class OTXSegmentationDataset(OTXDataset[SegDataEntity]):
    """OTXDataset class for segmentation task."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
    ) -> None:
        super().__init__(
            dm_subset,
            transforms,
            mem_cache_handler,
            mem_cache_img_max_size,
            max_refetch,
            image_color_channel,
            stack_images,
        )
        self.label_info = SegMetaInfo(
            label_names=self.label_info.label_names,
            label_groups=self.label_info.label_groups,
        )

    def _get_item_impl(self, index: int) -> SegDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        num_classes = self.label_info.num_classes
        ignored_labels: list[int] = []
        img_data, img_shape = self._get_img_data_and_shape(img)

        # create 2D class mask. We use np.sum() since Datumaro returns 3D masks (one for each class)
        mask_anns = np.sum(
            [ann.as_class_mask() for ann in item.annotations if isinstance(ann, Mask)],
            axis=0,
            dtype=np.uint8,
        )
        mask = torch.as_tensor(mask_anns, dtype=torch.long)
        # assign possible ignored labels from dataset to max label class + 1.
        # it is needed to compute mDice metric.
        mask[mask == 255] = num_classes
        entity = SegDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            gt_seg_map=tv_tensors.Mask(
                mask,
            ),
        )
        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect SegDataEntity into SegBatchDataEntity in data loader."""
        return SegBatchDataEntity.collate_fn
