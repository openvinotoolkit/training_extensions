# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXActionClsDataset."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import torch
from datumaro import Label

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.action_classification import ActionClsBatchDataEntity, ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo, VideoInfo
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER
from otx.core.types.image import ImageColorChannel

if TYPE_CHECKING:
    from datumaro import DatasetSubset

    from otx.core.data.dataset.base import Transforms
    from otx.core.data.mem_cache import MemCacheHandlerBase


class OTXActionClsDataset(OTXDataset[ActionClsDataEntity]):
    """OTXDataset class for action classification task."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.BGR,
        stack_images: bool = True,
        to_tv_image: bool = True,
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
        # TODO(Someone): ImageColorChannel is not used in action classification task
        # This task only supports BGR color format.
        # There should be implementation that links between ImageColorChannel and action classification task.
        if self.image_color_channel != ImageColorChannel.BGR:
            msg = "Action classification task only supports BGR color format."
            raise ValueError(msg)

    def _get_item_impl(self, idx: int) -> ActionClsDataEntity | None:
        item = self.dm_subset[idx]

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]

        entity = ActionClsDataEntity(
            video=item.media,
            image=[],
            img_info=ImageInfo(
                img_idx=idx,
                img_shape=(0, 0),
                ori_shape=(0, 0),
                image_color_channel=self.image_color_channel,
            ),
            video_info=VideoInfo(),
            labels=torch.as_tensor([ann.label for ann in label_anns]),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ActionClsDataEntity into ActionClsBatchDataEntity."""
        return partial(ActionClsBatchDataEntity.collate_fn, stack_images=self.stack_images)
