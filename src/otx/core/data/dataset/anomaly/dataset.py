"""Anomaly Classification Dataset."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Callable

import torch
from datumaro import DatasetSubset, Image

from otx.core.data.dataset.base import OTXDataset, Transforms
from otx.core.data.entity.anomaly import (
    AnomalyClassificationDataBatch,
    AnomalyClassificationDataItem,
)
from otx.core.data.entity.base import ImageInfo
from otx.core.data.mem_cache import MemCacheHandlerBase
from otx.core.types.image import ImageColorChannel
from otx.core.types.task import OTXTaskType


class AnomalyDataset(OTXDataset):
    """OTXDataset class for anomaly classification task."""

    def __init__(
        self,
        task_type: OTXTaskType,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = ...,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
    ) -> None:
        self.task_type = task_type
        super().__init__(
            dm_subset,
            transforms,
            mem_cache_handler,
            mem_cache_img_max_size,
            max_refetch,
            image_color_channel,
            stack_images,
        )

    def _get_item_impl(self, index: int) -> AnomalyClassificationDataItem | None:
        datumaro_item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = datumaro_item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)
        label: torch.LongTensor = (
            torch.tensor(0.0, dtype=torch.long) if "good" in datumaro_item.id else torch.tensor(1.0, dtype=torch.long)
        )

        item = AnomalyClassificationDataItem(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
            label=label,
        )
        return self._apply_transforms(item)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect SegDataEntity into SegBatchDataEntity in data loader."""
        return AnomalyClassificationDataBatch.collate_fn
