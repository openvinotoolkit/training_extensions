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
from collections import defaultdict
from typing import List, Optional

import numpy as np
from mmcv import ConfigDict

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.argument_checks import check_input_parameters_type


@check_input_parameters_type()
def find_label_by_name(labels: List[LabelEntity], name: str, domain: Domain):
    """Return label from name."""
    matching_labels = [label for label in labels if label.name == name]
    if len(matching_labels) == 1:
        return matching_labels[0]
    if len(matching_labels) == 0:
        label = LabelEntity(name=name, domain=domain, id=ID(int(name)))
        labels.append(label)
        return label
    raise ValueError("Found multiple matching labels")


def load_cls_annotations(ann_file, data_root):
    """Load annotation file to get video information."""
    video_infos = []
    with open(ann_file, "r", encoding="UTF-8") as fin:
        for line in fin:
            line_split = line.strip().split()
            video_info = {}
            idx = 0
            # idx for frame_dir
            if line_split[0] == "#":
                continue
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
def load_det_annotations(ann_file, data_root):
    """Load AVA annotations."""
    video_infos = []
    records_dict_by_img = defaultdict(list)
    with open(ann_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line_split = line.strip().split(",")

            label = int(line_split[6])
            video_id = line_split[0]
            timestamp = int(line_split[1])
            img_key = f"{video_id},{timestamp}"

            entity_box = np.array(list(map(float, line_split[2:6])))
            entity_id = int(line_split[7])

            video_info = dict(
                video_id=video_id,
                timestamp=timestamp,
                entity_box=entity_box,
                label=label,
                entity_id=entity_id,
            )
            records_dict_by_img[img_key].append(video_info)

    for img_key in records_dict_by_img:
        video_id, timestamp = img_key.split(",")
        bboxes, labels, entity_ids = parse_img_record(records_dict_by_img[img_key])
        ann = dict(gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids)
        frame_dir = video_id
        if data_root is not None:
            frame_dir = osp.join(data_root, frame_dir)
        # FIXME Image shape is hard-coded, this will be replaced with CVAT format
        video_info = dict(
            frame_dir=frame_dir,
            video_id=video_id,
            timestamp=int(timestamp),
            img_key=img_key,
            ann=ann,
            width=320,
            height=240,
        )
        video_infos.append(video_info)

    return video_infos


def parse_img_record(img_records):
    """Accumulate and colligate bbox annotation info."""
    bboxes, labels, entity_ids = [], [], []
    while len(img_records) > 0:
        img_record = img_records[0]
        num_img_records = len(img_records)

        selected_records = [x for x in img_records if np.array_equal(x["entity_box"], img_record["entity_box"])]

        num_selected_records = len(selected_records)
        img_records = [x for x in img_records if not np.array_equal(x["entity_box"], img_record["entity_box"])]

        assert len(img_records) + num_selected_records == num_img_records

        bboxes.append(img_record["entity_box"])
        valid_labels = np.array([selected_record["label"] for selected_record in selected_records])
        labels.append(valid_labels)
        entity_ids.append(img_record["entity_id"])

    bboxes = np.stack(bboxes)
    entity_ids = np.stack(entity_ids)
    return bboxes, labels, entity_ids


# pylint: disable=too-many-locals
def load_cls_dataset(
    ann_file_path: str,
    data_root_dir: str,
    domain: Domain,
    subset: Subset = Subset.NONE,
    labels_list: Optional[List[LabelEntity]] = None,
):
    """Convert video annotation information into DatasetItemEntity."""
    dataset_items = []
    video_infos = load_cls_annotations(ann_file_path, data_root_dir)

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


# pylint: disable=too-many-locals
def load_det_dataset(
    ann_file_path: str,
    data_root_dir: str,
    domain: Domain,
    subset: Subset = Subset.NONE,
    labels_list: Optional[List[LabelEntity]] = None,
):
    """Convert video annotation information into DatasetItemEntity."""
    dataset_items = []
    video_infos = load_det_annotations(ann_file_path, data_root_dir)

    for video_info in video_infos:
        ann = video_info.pop("ann")
        # TODO Check use of entity_ids
        gt_bboxes = ann["gt_bboxes"]
        gt_labels = ann["gt_labels"]
        shapes = []
        for bbox, labels in zip(gt_bboxes, gt_labels):
            labels = [find_label_by_name(labels_list, str(label), domain) for label in labels]
            shapes.append(
                Annotation(
                    Rectangle(bbox[0], bbox[1], bbox[2], bbox[3]),
                    [ScoredLabel(label, probability=1.0) for label in labels],
                )
            )
        dataset_item = DatasetItemEntity(
            media=ConfigDict(video_info),
            annotation_scene=AnnotationSceneEntity(annotations=shapes, kind=AnnotationSceneKind.ANNOTATION),
            subset=subset,
        )
        dataset_items.append(dataset_item)

    return dataset_items
