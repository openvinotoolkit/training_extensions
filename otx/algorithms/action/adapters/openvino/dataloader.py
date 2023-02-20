"""Data loaders for OpenVINO action recognition models."""

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

from copy import deepcopy
from typing import Dict, List

import numpy as np
from compression.api import DataLoader

from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.entities.datasets import DatasetEntity, DatasetItemEntity


def get_ovdataloader(dataset: DatasetEntity, task_type: str, clip_len: int, width: int, height: int) -> DataLoader:
    """Find proper dataloader for dataset and task type.

    If dataset has only a single video, this returns DataLoader for online demo
    If dataset has multiple videos, this return DataLoader for academia evaluation
    """
    if _is_multi_video(dataset):
        if task_type == "ACTION_CLASSIFICATION":
            return ActionOVClsDataLoader(dataset, clip_len, width, height)
        if task_type == "ACTION_DETECTION":
            return ActionOVDetDataLoader(dataset, clip_len, width, height)
        raise NotImplementedError(f"{task_type} is not supported from action task")
    return ActionOVDemoDataLoader(dataset, task_type, clip_len, width, height)


def _is_multi_video(dataset: DatasetEntity) -> bool:
    """Check dataset has multiple videos."""
    _video_id = dataset[0].get_metadata()[0].data.video_id
    for data in dataset:
        video_id = data.get_metadata()[0].data.video_id
        if _video_id != video_id:
            return True
    return False


class ActionOVDemoDataLoader(DataLoader):
    """DataLoader for online demo purpose.

    Since it is for online demo purpose it selects background frames from neighbor of key frame
    """

    def __init__(self, dataset: DatasetEntity, task_type: str, clip_len: int, width: int, height: int):
        self.task_type = task_type
        self.dataset = dataset
        self.clip_len = clip_len
        self.width = width
        self.height = height
        self.interval = 2

    def __len__(self):
        """Length of data loader."""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Sample frames from back and forth of key frame."""
        start = index - (self.clip_len // 2) * self.interval
        end = index + ((self.clip_len + 1) // 2) * self.interval
        frame_inds = list(range(start, end, self.interval))
        frame_inds = np.clip(frame_inds, 0, len(self.dataset) - 1)
        dataset_items = []
        for idx in frame_inds:
            dataset_item = self.dataset[int(idx)]
            dataset_items.append(dataset_item)
        return dataset_items

    def add_prediction(self, data: List[DatasetItemEntity], prediction: AnnotationSceneEntity):
        """Add prediction results to key frame.

        From sampling methods, we know that data[len(data) // 2] is key frame
        """
        dataset_item = data[len(data) // 2]
        if self.task_type == "ACTION_CLASSIFICATION":
            dataset_item.append_labels(prediction.annotations[0].get_labels())
        else:
            dataset_item.append_annotations(prediction.annotations)


class ActionOVClsDataLoader(DataLoader):
    """DataLoader for evaluation of action classification models.

    It iterates through clustered video, and it samples frames from given video
    """

    def __init__(self, dataset: DatasetEntity, clip_len: int, width: int, height: int):
        self.clip_len = clip_len
        self.width = width
        self.height = height

        video_info: Dict[str, List[DatasetItemEntity]] = {}
        for dataset_item in dataset:
            metadata = dataset_item.get_metadata()[0].data
            video_id = metadata.video_id
            if video_id in video_info:
                video_info[video_id].append(dataset_item)
            else:
                video_info[video_id] = [dataset_item]
        self.dataset = list(video_info.values())

        self.interval = 4

    def __len__(self):
        """Length of data loader."""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Sample frames from given video."""
        items = self.dataset[index]
        indices = self._get_indices(len(items))
        dataset_items = []
        for idx in indices:
            dataset_item = items[idx]
            dataset_items.append(dataset_item)
        return dataset_items

    def _get_indices(self, video_len: int):
        """Sample frame indices from video length."""
        ori_clip_len = self.clip_len * self.interval
        if video_len > ori_clip_len - 1:
            start = (video_len - ori_clip_len + 1) / 2
        else:
            start = 0
        frame_inds = np.arange(self.clip_len) * self.interval + int(start)
        frame_inds = np.mod(frame_inds, video_len)
        frame_inds = frame_inds.astype(np.int)
        return frame_inds

    def add_prediction(self, dataset: DatasetEntity, data: List[DatasetItemEntity], prediction: AnnotationSceneEntity):
        """Add prediction to dataset.

        Add prediction result to dataset_item in dataset, which has same video id with video data.
        """
        video_id = data[0].get_metadata()[0].data.video_id
        for dataset_item in dataset:
            if dataset_item.get_metadata()[0].data.video_id == video_id:
                dataset_item.append_labels(prediction.annotations[0].get_labels())


class ActionOVDetDataLoader(DataLoader):
    """DataLoader for evaluation of spatio-temporal action detection models.

    It iterates through DatasetEntity, which only contains non-empty frame(frame with actor annotation)
    It samples background frames from original DatasetEntity, which contain both empty frame and non-empty frame
    """

    def __init__(self, dataset: DatasetEntity, clip_len: int, width: int, height: int):
        self.original_dataset = dataset
        self.clip_len = clip_len
        self.width = width
        self.height = height

        self.dataset = deepcopy(dataset)
        video_info: Dict[str, Dict[str, int]] = {}
        for idx, dataset_item in enumerate(self.dataset):
            metadata = dataset_item.get_metadata()[0].data
            video_id = metadata.video_id
            timestamp = metadata.frame_idx
            if video_id in video_info:
                if video_info[video_id]["timestamp_start"] > timestamp:
                    video_info[video_id]["timestamp_start"] = timestamp
                if video_info[video_id]["timestamp_end"] < timestamp:
                    video_info[video_id]["timestamp_end"] = timestamp
            else:
                video_info[video_id] = {
                    "start_index": idx,
                    "timestamp_start": timestamp,
                    "timestamp_end": timestamp,
                }
        remove_indices = []
        for idx, dataset_item in enumerate(self.dataset):
            metadata = dataset_item.get_metadata()[0].data
            if metadata.is_empty_frame:
                remove_indices.append(idx)
                continue
            metadata.update("start_index", video_info[metadata.video_id]["start_index"])
            metadata.update("timestamp_start", video_info[metadata.video_id]["timestamp_start"])
            metadata.update("timestamp_end", video_info[metadata.video_id]["timestamp_end"])
        self.dataset.remove_at_indices(remove_indices)

        self.interval = 2
        self.fps = 1

    def __len__(self):
        """Length of data loader."""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Sample frames from back and forth of key frame, and all frames are from same video with key frame."""
        metadata = self.dataset[index].get_metadata()[0].data
        timestamp = metadata.frame_idx
        timestamp_start = metadata.timestamp_start
        timestamp_end = metadata.timestamp_end
        indices = self._get_indices(timestamp, timestamp_start, timestamp_end)
        indices = indices - timestamp_start + metadata.start_index
        dataset_items = []
        for idx in indices:
            dataset_item = self.original_dataset[int(idx)]
            dataset_items.append(dataset_item)
        return dataset_items

    def _get_indices(self, timestamp: int, timestamp_start: int, timestamp_end: int):
        """Get indices from timestamp.

        Samples from back and forth of key timestamp, and clips using start, and end timestamp
        """
        start = timestamp - (self.clip_len // 2) * self.interval
        end = timestamp + ((self.clip_len + 1) // 2) * self.interval
        frame_inds = list(range(start, end, self.interval))
        frame_inds = np.clip(frame_inds, timestamp_start, timestamp_end)
        return frame_inds

    def add_prediction(self, data: List[DatasetItemEntity], prediction: AnnotationSceneEntity):
        """Add prediction results to key frame."""
        dataset_item = data[len(data) // 2]
        dataset_item.append_annotations(prediction.annotations)
