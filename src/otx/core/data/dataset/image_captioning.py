# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset for Image-Caption pair."""

from __future__ import annotations

from functools import partial
from typing import Callable

from datumaro import Caption, Image

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.image_captioning import ImageCaptionBatchDataEntity, ImageCaptionDataEntity


class ImageCaptionDataset(OTXDataset[ImageCaptionDataEntity]):
    """Dataset for Image-Caption pair."""

    def _get_item_impl(self, index: int) -> ImageCaptionDataEntity | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        captions = []
        for ann in item.annotations:
            if isinstance(ann, Caption):
                captions.append(ann)
            else:
                # TODO(@harimkang): Need to Check Zero-shot Classification
                caption = Caption(caption=f"a photo of {ann.label}")
                if caption not in captions:
                    captions.append(caption)

        entity = ImageCaptionDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
            captions=[ann.caption for ann in captions],
        )
        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ImageCaptionDataEntity into ImageCaptionBatchDataEntity in data loader."""
        return partial(ImageCaptionBatchDataEntity.collate_fn, stack_images=self.stack_images)
