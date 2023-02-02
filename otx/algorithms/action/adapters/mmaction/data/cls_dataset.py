"""Base MMDataset for Action Recognition Task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from copy import copy
from typing import Any, Dict, List, Sequence

import numpy as np
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from mmaction.datasets.rawframe_dataset import RawframeDataset

from otx.algorithms.action.adapters.mmaction.data.pipelines import RawFrameDecode
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)


# pylint: disable=too-many-instance-attributes
@DATASETS.register_module()
class OTXActionClsDataset(RawframeDataset):
    """Wrapper that allows using a OTX dataset to train mmaction models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.
    """

    class _DataInfoProxy:
        def __init__(self, otx_dataset: DatasetEntity, labels: List[LabelEntity], modality: str):
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}
            self.modality = modality
            self.video_info: Dict[str, Any] = {}
            self._update_meta_data()

        def __len__(self) -> int:
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

        def __getitem__(self, index: int) -> Dict[str, Any]:
            """Prepare training data item.

            Action classification needs video for training, therefore this function generate item from video_info
            """

            item = self.video_info[list(self.video_info.keys())[index]]
            return item

    @check_input_parameters_type({"otx_dataset": DatasetParamTypeCheck})
    # pylint: disable=too-many-arguments, invalid-name, super-init-not-called
    # TODO Check need for additional params such as multi_class, with_offset
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
        self.test_mode = test_mode
        self.modality = modality

        self.video_infos = OTXActionClsDataset._DataInfoProxy(otx_dataset, labels, modality)

        self.pipeline = Compose(pipeline)
        for transform in self.pipeline.transforms:
            if isinstance(transform, RawFrameDecode):
                transform.otx_dataset = self.otx_dataset

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.video_infos)

    @check_input_parameters_type()
    def prepare_train_frames(self, idx: int) -> Dict[str, Any]:
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data
        """
        item = copy(self.video_infos[idx])  # Copying dict(), not contents
        return self.pipeline(item)

    @check_input_parameters_type()
    def prepare_test_frames(self, idx: int) -> Dict[str, Any]:
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data
        """
        item = copy(self.video_infos[idx])  # Copying dict(), not contents
        return self.pipeline(item)
