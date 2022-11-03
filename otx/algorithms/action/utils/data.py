"""Collection of utils for data in Action Task."""

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

import os.path as osp
from typing import List, Optional

# TODO Move find_label_by_name to common
from otx.algorithms.detection.utils.data import find_label_by_name
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset


def load_annotations(ann_file, data_root):
    """Load annotation file to get video information."""
    video_infos = []
    with open(ann_file, "r", encoding="UTF-8") as fin:
        for line in fin:
            line_split = line.strip().split()
            video_info = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            if data_root is not None:
                frame_dir = osp.join(data_root, frame_dir)
            video_info["frame_dir"] = frame_dir
            idx += 1
            # idx for total_frames
            # TODO Support offsets in dataset
            video_info["total_frames"] = int(line_split[idx])
            idx += 1
            # idx for label[s]
            # TODO Support multi-label setting
            label = [int(x) for x in line_split[idx:]]
            assert label, f"missing label in line: {line}"
            assert len(label) == 1
            video_info["label"] = label[0]
            video_infos.append(video_info)

    return video_infos


# pylint: disable=too-many-locals
def load_rawframe_dataset(
    ann_file_path: str,
    data_root_dir: str,
    domain: Domain,
    subset: Subset = Subset.NONE,
    labels_list: Optional[List[LabelEntity]] = None,
):
    """Convert video annotation information into DatasetItemEntity."""
    dataset_items = []
    video_infos = load_annotations(ann_file_path, data_root_dir)

    for video_info in video_infos:
        label = video_info.pop("label")
        label = find_label_by_name(labels_list, str(label), domain)
        shapes = [Annotation(Rectangle.generate_full_box(), [ScoredLabel(label)])]
        dataset_item = DatasetItemEntity(
            media=video_info,
            annotation_scene=AnnotationSceneEntity(annotations=shapes, kind=AnnotationSceneKind.ANNOTATION),
            subset=subset,
        )
        dataset_items.append(dataset_item)

    return dataset_items
