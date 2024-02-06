"""Anomaly Classification Dataset."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Callable

import torch
from datumaro import Image

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.anomaly import (
    AnomalyClassificationDataBatch,
    AnomalyClassificationDataItem,
)
from otx.core.data.entity.base import ImageInfo


class AnomalyClassificationDataset(OTXDataset[AnomalyClassificationDataItem]):
    """OTXDataset class for anomaly classification task."""

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
