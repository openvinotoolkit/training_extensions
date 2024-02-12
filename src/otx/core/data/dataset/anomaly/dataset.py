"""Anomaly Classification Dataset."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from anomalib.data.utils import masks_to_boxes
from datumaro import DatasetSubset, Image
from torchvision import io

from otx.core.data.dataset.base import OTXDataset, Transforms
from otx.core.data.entity.anomaly import (
    AnomalyClassificationDataBatch,
    AnomalyClassificationDataItem,
    AnomalyDetectionDataBatch,
    AnomalyDetectionDataItem,
    AnomalySegmentationDataBatch,
    AnomalySegmentationDataItem,
)
from otx.core.data.entity.base import ImageInfo
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.types.image import ImageColorChannel
from otx.core.types.task import OTXTaskType


class AnomalyDataset(OTXDataset):
    """OTXDataset class for anomaly classification task."""

    def __init__(
        self,
        task_type: OTXTaskType,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
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

    def _get_item_impl(self, index: int) -> AnomalyClassificationDataItem | AnomalySegmentationDataBatch:
        datumaro_item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = datumaro_item.media_as(Image)
        # returns image in RGB format if self.image_color_channel is RGB
        img_data, img_shape = self._get_img_data_and_shape(img)
        # Note: This assumes that the dataset is in MVTec format.
        # We can't use datumaro label id as it returns some number like 3 for good from which it is hard to infer
        # whether the image is Anomalous or Normal. Because it leads to other questions like what do numbers 0,1,2 mean?
        label: torch.LongTensor = (
            torch.tensor(0.0, dtype=torch.long) if "good" in datumaro_item.id else torch.tensor(1.0, dtype=torch.long)
        )
        item: AnomalyClassificationDataItem | AnomalySegmentationDataItem | AnomalyDetectionDataItem
        if self.task_type == OTXTaskType.ANOMALY_CLASSIFICATION:
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
        elif self.task_type == OTXTaskType.ANOMALY_SEGMENTATION:
            # Note: this part of code is brittle. Ideally Datumaro should return masks
            # Another major problem with this is that it assumes that the dataset passed is in MVTec format
            mask_file_path = (
                Path("/".join(datumaro_item.media.path.split("/")[:-3]))
                / "ground_truth"
                / f"{('/'.join(datumaro_item.media.path.split('/')[-2:])).replace('.png','_mask.png')}"
            )
            mask = torch.zeros(1, img_shape[0], img_shape[1], dtype=torch.uint8)
            if mask_file_path.exists():
                # read and convert to binary mask
                mask = (io.read_image(str(mask_file_path), mode=io.ImageReadMode.GRAY) / 255).to(torch.uint8)
            item = AnomalySegmentationDataItem(
                image=img_data,
                img_info=ImageInfo(
                    img_idx=index,
                    img_shape=img_shape,
                    ori_shape=img_shape,
                    image_color_channel=self.image_color_channel,
                ),
                label=label,
                mask=mask,
            )
        elif self.task_type == OTXTaskType.ANOMALY_DETECTION:
            # Note: this part of code is brittle. Ideally Datumaro should return masks
            mask_file_path = (
                Path("/".join(datumaro_item.media.path.split("/")[:-3]))
                / "ground_truth"
                / f"{('/'.join(datumaro_item.media.path.split('/')[-2:])).replace('.png','_mask.png')}"
            )
            mask = torch.zeros(1, img_shape[0], img_shape[1], dtype=torch.uint8)
            if mask_file_path.exists():
                # read and convert to binary mask
                mask = (io.read_image(str(mask_file_path), mode=io.ImageReadMode.GRAY) / 255).to(torch.uint8)
            boxes, _ = masks_to_boxes(mask)
            item = AnomalyDetectionDataItem(
                image=img_data,
                img_info=ImageInfo(
                    img_idx=index,
                    img_shape=img_shape,
                    ori_shape=img_shape,
                    image_color_channel=self.image_color_channel,
                ),
                label=label,
                boxes=boxes[0],
                # mask is used for pixel-level metric computation. We can't assume that this will always be available
                mask=mask,
            )
        else:
            msg = f"Task {self.task_type} is not supported yet."
            raise NotImplementedError(msg)
        # without ignore the following error is returned
        # Incompatible return value type (got "Any | None", expected
        # "AnomalyClassificationDataItem | AnomalySegmentationDataBatch")
        return self._apply_transforms(item)  # type: ignore[return-value]

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect SegDataEntity into SegBatchDataEntity in data loader."""
        if self.task_type == OTXTaskType.ANOMALY_CLASSIFICATION:
            return AnomalyClassificationDataBatch.collate_fn
        if self.task_type == OTXTaskType.ANOMALY_SEGMENTATION:
            return AnomalySegmentationDataBatch.collate_fn
        if self.task_type == OTXTaskType.ANOMALY_DETECTION:
            return AnomalyDetectionDataBatch.collate_fn
        msg = f"Task {self.task_type} is not supported yet."
        raise NotImplementedError(msg)
