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
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            data_info = dict(
                dataset_item=item,
                index=index,
                ann_info=dict(label_list=self.labels),
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
        data_prefix=str,
        test_mode: bool = False,
        filename_tmpl: str = "img_{:05}.jpg",
        start_index: int = 1,
        modality: str = "RGB",
    ):
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        self.modality = modality

        self.data_infos = OTXRawframeDataset._DataInfoProxy(otx_dataset, labels)

        self.pipeline = Compose(pipeline)
        self.make_video_infos()

    def __len__(self):
        """Return length of dataset."""
        return len(self.data_infos)

    @check_input_parameters_type()
    def prepare_train_frames(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        :param idx: int, Index of data.
        :return dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """
        item = copy(self.data_infos[idx])  # Copying dict(), not contents
        self.pre_pipeline(item)
        return self.pipeline(item)

    @check_input_parameters_type()
    def prepare_test_frames(self, idx: int) -> dict:
        """Get testing data after pipeline.

        :param idx: int, Index of data.
        :return dict: Testing data after pipeline with new keys introduced by pipeline.
        """
        item = copy(self.data_infos[idx])  # Copying dict(), not contents
        self.pre_pipeline(item)
        return self.pipeline(item)

    @check_input_parameters_type()
    def pre_pipeline(self, results: Dict[str, Any]):
        """Prepare results dict for pipeline. Add expected keys to the dict."""
        # Should we make function like LoadImageFromOTXDataset ?
        results["frame_dir"] = results["dataset_item"].media["frame_dir"]
        results["total_frames"] = results["dataset_item"].media["total_frames"]
        results["label"] = results["dataset_item"].media["label"]
        results["filename_tmpl"] = self.filename_tmpl
        results["modality"] = self.modality
        results["start_index"] = self.start_index

    def make_video_infos(self):
        """Make mmaction style video infos."""
        self.video_infos = []
        for data_info in self.data_infos:
            media = data_info["dataset_item"].media
            annotation = data_info["dataset_item"].get_annotations()
            if len(annotation) == 0:
                label = None
            else:
                label = int(data_info["dataset_item"].get_roi_labels(self.labels)[0].id)
            media["label"] = label
            self.video_infos.append(media)
