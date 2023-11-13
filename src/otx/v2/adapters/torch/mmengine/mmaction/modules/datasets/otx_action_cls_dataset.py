"""Base Dataset for Action Recognition Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from mmaction.datasets.rawframe_dataset import RawframeDataset
from mmaction.registry import DATASETS
from mmengine.dataset import Compose

from otx.v2.adapters.torch.mmengine.mmaction.modules.datasets.pipelines import OTXRawFrameDecode
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.label import LabelEntity


@DATASETS.register_module()
class OTXActionClsDataset(RawframeDataset):
    """Wrapper that allows using a OTX dataset to train mmaction models."""
    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: list[LabelEntity],
        pipeline: Sequence[dict],
        test_mode: bool = False,
        modality: str = "RGB",  # [RGB, FLOW(Optical flow)]
    ) -> None:
        """OTXActionClassificationDataset.

        Args:
            otx_dataset (DatasetEntity): DatasetEntity that includes the img, annotation, meta.
            labels (list[LabelEntity]): LabelEntitities that include the label information.
            pipeline (Sequence[dict]): A sequence of data transforms.
            test_mode (bool, optional): Store True when building test or validation dataset. Defaults to False.
            modality (str, optional): Modality of data. Support ``RGB``, ``Flow``. Defaults to "RGB".
        """
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.label_idx = {label.id: i for i, label in enumerate(labels)}
        self.CLASSES = [label.name for label in labels]
        self.test_mode = test_mode
        self.modality = modality
        self.video_info: dict[str, Any] = {}
        self._update_meta_data()

        self.pipeline = Compose(pipeline)
        for transform in self.pipeline.transforms:
            if isinstance(transform, OTXRawFrameDecode):
                transform.otx_dataset = self.otx_dataset

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Prepare training data item.

        Action classification needs video for training, therefore this function generate item from video_info
        """
        item = self.video_info[list(self.video_info.keys())[index]]
        return self.pipeline(item)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.video_info)

    def _update_meta_data(self) -> None:
        """Update video metadata of each item in self.otx_dataset."""
        video_info: dict = {}
        start_index = 0
        for idx, item in enumerate(self.otx_dataset):
            metadata = item.get_metadata()[0].data

            if metadata.video_id not in video_info:

                label = int(item.get_roi_labels(self.labels)[0].id) if len(item.get_annotations()) > 0 else None
                ignored_labels = np.array([self.label_idx[label.id] for label in item.ignored_labels])
                video_info[metadata.video_id] = {
                    "total_frames": 1,
                    "start_index": idx,
                    "label": label,
                    "ignored_labels": ignored_labels,
                    "modality": self.modality,
                }
                start_index = idx
            else:
                video_info[metadata.video_id]["total_frames"] += 1
                video_info[metadata.video_id]["start_index"] = start_index

        self.video_info.update(video_info)
