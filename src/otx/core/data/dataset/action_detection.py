# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXActionDetDataset."""

from __future__ import annotations

import pickle
from pathlib import Path
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
                image_color_channel=self.image_color_channel,
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
            proposals=self._get_proposals(
                item.media.path,
                self.dm_subset.infos().get(f"{self.dm_subset.name}_proposals", None),
            ),
        )

        return self._apply_transforms(entity)

    @staticmethod
    def _get_proposals(frame_path: str, proposal_file: str | None) -> np.ndarray:
        """Get proposal from frame path and proposal file name.

        Datumaro AVA dataset expect data structure as
        - data_root/
            - frames/
                - video0
                    - video0_0001.jpg
                    - vdieo0_0002.jpg
            - annotations/
                - train.csv
                - val.csv
                - train.pkl
                - val.pkl
        """
        if proposal_file is None:
            return np.array([[0, 0, 1, 1]], dtype=np.float64)

        annotation_dir = Path(frame_path).parent.parent.parent
        proposal_file_path = annotation_dir / "annotations" / proposal_file
        if not proposal_file_path.exists():
            return np.array([[0, 0, 1, 1]], dtype=np.float64)
        with Path.open(proposal_file_path, "rb") as f:
            info = pickle.load(f)  # noqa: S301
            return (
                info[",".join(Path(frame_path).stem.rsplit("_", 1))][:, :4]
                if ",".join(Path(frame_path).stem.rsplit("_", 1)) in info
                else np.array([[0, 0, 1, 1]], dtype=np.float32)
            )

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ActionClsDataEntity into ActionClsBatchDataEntity."""
        return ActionDetBatchDataEntity.collate_fn
