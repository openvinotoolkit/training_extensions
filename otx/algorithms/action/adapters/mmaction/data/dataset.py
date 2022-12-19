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
from typing import List, Sequence

import numpy as np
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from mmaction.datasets.rawframe_dataset import RawframeDataset

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)


# pylint: disable=too-many-instance-attributes
@DATASETS.register_module()
class OTXRawframeDataset(RawframeDataset):
    """Wrapper that allows using a OTX dataset to train mmaction models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.
    """

    class _DataInfoProxy:
        def __init__(self, otx_dataset, labels):
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}

        def __len__(self):
            return len(self.otx_dataset)

        def __getitem__(self, index):
            """Prepare a dict 'data_info' that is expected by the mmaction pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """

            dataset = self.otx_dataset
            item = dataset[index]
            video = item.media["video"]
            total_frames = item.media["total_frames"]
            start_index = item.media["start_index"]
            if len(item.get_annotations()) == 0:
                label = None
            else:
                label = int(item.get_roi_labels(self.labels)[0].id)
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            data_info = dict(
                data=video,
                label=label,
                index=index,
                ann_info=dict(label_list=self.labels),
                total_frames=total_frames,
                start_index=start_index,
                ignored_labels=ignored_labels,
            )

            return data_info

    @check_input_parameters_type({"otx_dataset": DatasetParamTypeCheck})
    # pylint: disable=too-many-arguments, invalid-name, super-init-not-called
    # TODO Check need for additional params such as multi_class, with_offset
    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: List[LabelEntity],
        pipeline: Sequence[dict],
        test_mode: bool = False,
        filename_tmpl: str = "img_{:05}.jpg",
        start_index: int = 1,
        modality: str = "RGB",
    ):
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.test_mode = test_mode
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        self.modality = modality

        self.video_infos = OTXRawframeDataset._DataInfoProxy(otx_dataset, labels)

        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Return length of dataset."""
        return len(self.video_infos)

    @check_input_parameters_type()
    def prepare_train_frames(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        :param idx: int, Index of data.
        :return dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """
        item = copy(self.video_infos[idx])  # Copying dict(), not contents
        item["modality"] = self.modality
        return self.pipeline(item)

    @check_input_parameters_type()
    def prepare_test_frames(self, idx: int) -> dict:
        """Get testing data after pipeline.

        :param idx: int, Index of data.
        :return dict: Testing data after pipeline with new keys introduced by pipeline.
        """
        item = copy(self.video_infos[idx])  # Copying dict(), not contents
        item["modality"] = self.modality
        return self.pipeline(item)
