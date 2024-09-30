# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX diffusion dataset."""

from __future__ import annotations

from functools import partial
from typing import Callable

from datumaro import Image

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.diffusion import DiffusionBatchDataEntity, DiffusionDataEntity


class OTXDiffusionDataset(OTXDataset[DiffusionDataEntity]):
    """Diffusion dataset class."""

    def _get_item_impl(self, idx: int) -> DiffusionDataEntity | None:
        item = self.dm_subset[idx]
        caption = item.annotations[0].caption
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)
        entity = DiffusionDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=idx,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
            caption=caption,
        )
        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect OTXDataEntity into OTXBatchDataEntity in data loader."""
        return partial(
            DiffusionBatchDataEntity.collate_fn,
            stack_images=self.stack_images,
        )
