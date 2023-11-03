"""Base Dataset for Action Recognition Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, List, Sequence

import numpy as np
from mmaction.registry import DATASETS
from mmengine.dataset import Compose
from mmaction.datasets.rawframe_dataset import RawframeDataset

from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.label import LabelEntity
from otx.v2.adapters.torch.mmengine.mmaction.modules.datasets.pipelines import RawFrameDecode


# pylint: disable=too-many-instance-attributes
@DATASETS.register_module()
class OTXActionClsDataset(RawframeDataset):
    """Wrapper that allows using a OTX dataset to train mmaction models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.
    """
    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: List[LabelEntity],
        pipeline: Sequence[dict],
        test_mode: bool = False,
        modality: str = "RGB",  # [RGB, FLOW(Optical flow)]
    ):
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.label_idx = {label.id: i for i, label in enumerate(labels)}
        self.CLASSES = [label.name for label in labels]
        self.test_mode = test_mode
        self.modality = modality
        self.video_info: Dict[str, Any] = {}
        self._update_meta_data()

        breakpoint()
        self.pipeline = Compose(pipeline)
        for transform in self.pipeline.transforms:
            if isinstance(transform, RawFrameDecode):
                transform.otx_dataset = self.otx_dataset

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Prepare training data item.

        Action classification needs video for training, therefore this function generate item from video_info
        """

        item = self.video_info[list(self.video_info.keys())[index]]
        return self.pipeline(item)
    
    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.video_info)

    def _update_meta_data(self):
        """Update video metadata of each item in self.otx_dataset.

        This function assumes that DatasetItemEntities in DatasetEntity are sorted by video id and frame idx
        During iterating DatasetsetEntitiy, this fucnction generates video_info(dictionary)
        video_info records metadata for each video, and it contains
            - total_frame: Total frame number of the video, this value will be used to sample frames for training
            - start_index: Offset for the video, this value will be added to sampled frame indices for the video
            - label: Action category of the video
            - modality = Modality of data, 'RGB' or 'Flow(Optical Flow)'
        """
        video_info = {}
        start_index = 0
        for idx, item in enumerate(self.otx_dataset):
            metadata = item.get_metadata()[0].data
            if metadata.video_id in video_info:
                video_info[metadata.video_id]["total_frames"] += 1
                video_info[metadata.video_id]["start_index"] = start_index
            else:
                if len(item.get_annotations()) > 0:
                    label = int(item.get_roi_labels(self.labels)[0].id)
                else:
                    label = None
                ignored_labels = np.array([self.label_idx[label.id] for label in item.ignored_labels])
                video_info[metadata.video_id] = {
                    "total_frames": 1,
                    "start_index": idx,
                    "label": label,
                    "ignored_labels": ignored_labels,
                    "modality": self.modality,
                }
                start_index = idx

        self.video_info.update(video_info)