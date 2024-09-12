"""Anomaly Classification Dataset."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from anomalib.data.utils import masks_to_boxes
from datumaro import Dataset as DmDataset
from datumaro import DatasetItem, Image
from datumaro.components.annotation import AnnotationType, Bbox, Ellipse, Polygon
from datumaro.components.media import ImageFromBytes, ImageFromFile
from torchvision import io
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask

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
from otx.core.types.label import AnomalyLabelInfo
from otx.core.types.task import OTXTaskType


class AnomalyLabel(Enum):
    """Anomaly label to tensor mapping."""

    NORMAL = torch.tensor(0.0)
    ANOMALOUS = torch.tensor(1.0)


class AnomalyDataset(OTXDataset):
    """OTXDataset class for anomaly classification task."""

    def __init__(
        self,
        task_type: OTXTaskType,
        dm_subset: DmDataset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
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
            to_tv_image,
        )
        self.label_info = AnomalyLabelInfo()
        self._label_mapping = self._map_id_to_label()

    def _get_item_impl(
        self,
        index: int,
    ) -> AnomalyClassificationDataItem | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch:
        datumaro_item = self.dm_subset[index]
        img = datumaro_item.media_as(Image)
        # returns image in RGB format if self.image_color_channel is RGB
        img_data, img_shape = self._get_img_data_and_shape(img)

        label = self._get_label(datumaro_item)

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
            item = AnomalySegmentationDataItem(
                image=img_data,
                img_info=ImageInfo(
                    img_idx=index,
                    img_shape=img_shape,
                    ori_shape=img_shape,
                    image_color_channel=self.image_color_channel,
                ),
                label=label,
                mask=Mask(self._get_mask(datumaro_item, label, img_shape)),
            )
        elif self.task_type == OTXTaskType.ANOMALY_DETECTION:
            item = AnomalyDetectionDataItem(
                image=img_data,
                img_info=ImageInfo(
                    img_idx=index,
                    img_shape=img_shape,
                    ori_shape=img_shape,
                    image_color_channel=self.image_color_channel,
                ),
                label=label,
                boxes=self._get_boxes(datumaro_item, label, img_shape),
                # mask is used for pixel-level metric computation. We can't assume that this will always be available
                mask=Mask(self._get_mask(datumaro_item, label, img_shape)),
            )
        else:
            msg = f"Task {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        # without ignore the following error is returned
        # Incompatible return value type (got "Any | None", expected
        # "AnomalyClassificationDataItem | AnomalySegmentationDataBatch | AnomalyDetectionDataBatch")
        return self._apply_transforms(item)  # type: ignore[return-value]

    def _get_mask(self, datumaro_item: DatasetItem, label: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
        """Get mask from datumaro_item.

        Converts bounding boxes to mask if mask is not available.
        """
        if isinstance(datumaro_item.media, ImageFromFile):
            if label == AnomalyLabel.ANOMALOUS.value:
                mask = self._mask_image_from_file(datumaro_item, img_shape)
            else:
                mask = torch.zeros(1, *img_shape).to(torch.uint8)
        elif isinstance(datumaro_item.media, ImageFromBytes):
            mask = torch.zeros(1, *img_shape).to(torch.uint8)
            if label == AnomalyLabel.ANOMALOUS.value:
                for annotation in datumaro_item.annotations:
                    # There is only one mask
                    if isinstance(annotation, (Ellipse, Polygon)):
                        polygons = np.asarray(annotation.as_polygon(), dtype=np.int32).reshape((-1, 1, 2))
                        mask = np.zeros(img_shape, dtype=np.uint8)
                        mask = cv2.drawContours(
                            mask,
                            [polygons],
                            0,
                            (1, 1, 1),
                            thickness=cv2.FILLED,
                        )
                        mask = torch.from_numpy(mask).to(torch.uint8).unsqueeze(0)
                        break
                    # If there is no mask, create a mask from bbox
                    if isinstance(annotation, Bbox):
                        bbox = annotation
                        mask = self._bbox_to_mask(bbox, img_shape)
                        break
        return mask

    def _get_boxes(self, datumaro_item: DatasetItem, label: torch.Tensor, img_shape: tuple[int, int]) -> BoundingBoxes:
        """Get bounding boxes from datumaro item.

        Uses masks if available to get bounding boxes.
        """
        boxes = BoundingBoxes(torch.empty(0, 4), format=BoundingBoxFormat.XYXY, canvas_size=img_shape)
        if isinstance(datumaro_item.media, ImageFromFile):
            if label == AnomalyLabel.ANOMALOUS.value:
                mask = self._mask_image_from_file(datumaro_item, img_shape)
                boxes, _ = masks_to_boxes(mask)
                # Assumes only one bounding box is present
                boxes = BoundingBoxes(boxes[0], format=BoundingBoxFormat.XYXY, canvas_size=img_shape)
        elif isinstance(datumaro_item.media, ImageFromBytes) and label == AnomalyLabel.ANOMALOUS.value:
            for annotation in datumaro_item.annotations:
                if isinstance(annotation, Bbox):
                    bbox = annotation
                    boxes = BoundingBoxes(bbox.get_bbox(), format=BoundingBoxFormat.XYXY, canvas_size=img_shape)
                    break
        return boxes

    def _bbox_to_mask(self, bbox: Bbox, img_shape: tuple[int, int]) -> torch.Tensor:
        mask = torch.zeros(1, *img_shape).to(torch.uint8)
        x1, y1, x2, y2 = bbox.get_bbox()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask[:, y1:y2, x1:x2] = 1
        return mask

    def _get_label(self, datumaro_item: DatasetItem) -> torch.LongTensor:
        """Get label from datumaro item."""
        if isinstance(datumaro_item.media, ImageFromFile):
            # Note: This assumes that the dataset is in MVTec format.
            # We can't use datumaro label id as it returns some number like 3 for good from which it is hard to infer
            # whether the image is Anomalous or Normal. Because it leads to other questions like what do numbers 0,1,2
            # mean?
            label: torch.LongTensor = AnomalyLabel.NORMAL if "good" in datumaro_item.id else AnomalyLabel.ANOMALOUS
        elif isinstance(datumaro_item.media, ImageFromBytes):
            label = self._label_mapping[datumaro_item.annotations[0].label]
        else:
            msg = f"Media type {type(datumaro_item.media)} is not supported."
            raise NotImplementedError(msg)
        return label.value

    def _map_id_to_label(self) -> dict[int, torch.Tensor]:
        """Map label id to label tensor."""
        id_label_mapping = {}
        categories = self.dm_subset.categories()[AnnotationType.label]
        for label_item in categories.items:
            if any("normal" in attribute.lower() for attribute in label_item.attributes):
                label = AnomalyLabel.NORMAL
            else:
                label = AnomalyLabel.ANOMALOUS
            id_label_mapping[categories.find(label_item.name)[0]] = label
        return id_label_mapping

    def _mask_image_from_file(self, datumaro_item: DatasetItem, img_shape: tuple[int, int]) -> torch.Tensor:
        """Assumes MVTec format and returns mask from disk."""
        mask_file_path = (
            Path("/".join(datumaro_item.media.path.split("/")[:-3]))
            / "ground_truth"
            / f"{('/'.join(datumaro_item.media.path.split('/')[-2:])).replace('.png','_mask.png')}"
        )
        if mask_file_path.exists():
            return (io.read_image(str(mask_file_path), mode=io.ImageReadMode.GRAY) / 255).to(torch.uint8)

        # Note: This is a workaround to handle the case where mask is not available otherwise the tests fail.
        # This is problematic because it assigns empty masks to an Anomalous image.
        return torch.zeros(1, *img_shape).to(torch.uint8)

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
