# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXActionDetDataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from datumaro import Bbox, Image
from datumaro.components.annotation import AnnotationType
from torchvision import tv_tensors

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.action_detection import ActionDetBatchDataEntity, ActionDetDataEntity
from otx.core.data.entity.base import ImageInfo


class OTXActionDetDataset(OTXDataset[ActionDetDataEntity]):
    """OTXDataset class for action detection task."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = len(self.dm_subset.categories()[AnnotationType.label])

    def _get_item_impl(self, idx: int) -> ActionDetDataEntity | None:
        item = self.dm_subset.get(id=self.ids[idx], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]
        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        entity = ActionDetDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=idx,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            labels=torch.nn.functional.one_hot(
                torch.as_tensor([ann.label for ann in bbox_anns]),
                self.num_classes,
            ).to(torch.float),
            frame_path=item.media.path,
            proposal_file=self.dm_subset.infos().get(f"{self.dm_subset.name}_proposals", None),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ActionClsDataEntity into ActionClsBatchDataEntity."""
        return ActionDetBatchDataEntity.collate_fn
