"""Collection of utils for dataset in Segmentation Task."""

# Copyright (C) 2021 Intel Corporation
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

import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import tqdm
from mmseg.datasets.custom import CustomDataset
from skimage.segmentation import felzenszwalb

from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.subset import Subset
from otx.api.utils.argument_checks import (
    DirectoryPathCheck,
    JsonFilePathCheck,
    OptionalDirectoryPathCheck,
    check_input_parameters_type,
)

logger = get_logger()

# pylint: disable=too-many-locals


@check_input_parameters_type({"annot_path": JsonFilePathCheck})
def get_classes_from_annotation(annot_path):
    """Getter function of classes from annotation."""
    with open(annot_path, encoding="UTF-8") as input_stream:
        content = json.load(input_stream)
        labels_map = content["labels_map"]

        categories = [(v["name"], v["id"]) for v in sorted(labels_map, key=lambda tup: int(tup["id"]))]

    return categories


@check_input_parameters_type({"value": OptionalDirectoryPathCheck})
def abs_path_if_valid(value):
    """Valid function of abs_path."""
    if value:
        return os.path.abspath(value)
    return None


@check_input_parameters_type()
def create_annotation_from_hard_seg_map(hard_seg_map: np.ndarray, labels: List[LabelEntity]):
    """Creation function from hard seg_map."""
    height, width = hard_seg_map.shape[:2]
    unique_labels = np.unique(hard_seg_map)

    annotations: List[Annotation] = []
    for label_id in unique_labels:
        label_id_entity = ID(f"{label_id:08}")
        matches = [label for label in labels if label.id == label_id_entity]
        if len(matches) == 0:
            continue

        assert len(matches) == 1
        label = matches[0]

        label_mask = hard_seg_map == label_id
        label_index_map = label_mask.astype(np.uint8)

        contours, hierarchies = cv2.findContours(label_index_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchies is None:
            continue

        for contour, hierarchy in zip(contours, hierarchies[0]):
            if hierarchy[3] != -1:
                continue

            contour = list(contour)
            if len(contour) <= 2:
                continue

            points = [Point(x=point[0][0] / width, y=point[0][1] / height) for point in contour]

            annotations.append(
                Annotation(
                    Polygon(points=points),
                    labels=[ScoredLabel(label)],
                    id=ID(f"{label_id:08}"),
                )
            )

    return annotations


@check_input_parameters_type({"ann_dir": OptionalDirectoryPathCheck})
def load_labels_from_annotation(ann_dir):
    """Load labels function from annotation."""
    if ann_dir is None:
        return []

    labels_map_path = os.path.join(ann_dir, "meta.json")
    labels = get_classes_from_annotation(labels_map_path)

    return labels


@check_input_parameters_type()
def add_labels(cur_labels: List[LabelEntity], new_labels: List[tuple]):
    """Add labels function."""
    for label_name, label_id in new_labels:
        matching_labels = [label for label in cur_labels if label.name == label_name]
        if len(matching_labels) > 1:
            raise ValueError("Found multiple matching labels")
        if len(matching_labels) == 0:
            label_id = label_id if label_id is not None else len(cur_labels)
            label = LabelEntity(name=label_name, domain=Domain.SEGMENTATION, id=ID(f"{label_id:08}"))
            cur_labels.append(label)


@check_input_parameters_type()
def check_labels(cur_labels: List[LabelEntity], new_labels: List[tuple]):
    """Check labels function."""
    cur_names = {label.name for label in cur_labels}
    new_names = {label[0] for label in new_labels}
    if cur_names != new_names:
        raise ValueError("Class names don't match from file to file")


@check_input_parameters_type()
def get_extended_label_names(labels: List[LabelEntity]):
    """Getter function of extended label names."""
    target_labels = [v.name for v in sorted(labels, key=lambda x: x.id)]
    all_labels = ["background"] + target_labels
    return all_labels


def get_valid_label_mask_per_batch(img_metas, num_classes):
    """Get valid label mask removing ignored classes to zero mask in a batch."""
    valid_label_mask_per_batch = []
    for _, meta in enumerate(img_metas):
        valid_label_mask = torch.Tensor([1 for _ in range(num_classes)])
        if "ignored_labels" in meta and meta["ignored_labels"]:
            valid_label_mask[meta["ignored_labels"]] = 0
        valid_label_mask_per_batch.append(valid_label_mask)
    return valid_label_mask_per_batch


@check_input_parameters_type()
def create_pseudo_masks(ann_file_path: str, data_root_dir: str, mode="FH"):
    """Create pseudo masks for Self-SL using DetCon."""
    if not os.path.isdir(ann_file_path):
        logger.info(
            (
                f"Creating pseudo masks with mode={mode} is required. "
                f"It may take some time. Once this process has been performed, "
                f"there is no need to proceed again with "
                f"ann_file_path={ann_file_path} and data_root_dir={data_root_dir}."
            )
        )
        os.makedirs(ann_file_path, exist_ok=False)
        img_list = os.listdir(data_root_dir)
        total_labels = []
        # create pseudo masks
        for path in tqdm.tqdm(img_list, total=len(img_list)):
            save_path = path.replace(".jpg", ".png") if path.endswith(".jpg") else path
            img = cv2.imread(os.path.join(data_root_dir, path))[..., ::-1]
            if mode == "FH":
                pseudo_mask = felzenszwalb(img, scale=1000, min_size=1000)
            else:
                raise ValueError(
                    (f"{mode} is not supported to create pseudo masks for DetCon." 'Choose one of ["FH"].')
                )
            cv2.imwrite(os.path.join(ann_file_path, save_path), pseudo_mask.astype(np.uint8))

            # get labels to create meta.json
            labels = np.unique(pseudo_mask)
            for label in labels:
                if label not in total_labels:
                    total_labels.append(label)

        # create meta.json
        # TODO (sungchul): to be updated as max(total_labels) -> max(total_labels)+1
        # Currently, background class is automatically added in the backend.
        # Considering background class, labels_map in meta.json should have one less than the number of labels.
        # If we don't need to consider background class, it will be updated to consider all labels.
        meta = {"labels_map": [{"name": f"target{i+1}", "id": i + 1} for i in range(max(total_labels))]}
        with open(os.path.join(ann_file_path, "meta.json"), "w", encoding="UTF-8") as f:
            json.dump(meta, f, indent=4)


@check_input_parameters_type({"ann_file_path": DirectoryPathCheck, "data_root_dir": DirectoryPathCheck})
def load_dataset_items(
    ann_file_path: str,
    data_root_dir: str,
    subset: Subset = Subset.NONE,
    labels_list: Optional[List[LabelEntity]] = None,
):
    """Load dataset items."""
    if "detcon" in ann_file_path:  # TODO (sungchul): deterministic condition
        create_pseudo_masks(ann_file_path, data_root_dir)

    ann_dir = abs_path_if_valid(ann_file_path)
    img_dir = abs_path_if_valid(data_root_dir)

    annot_labels = load_labels_from_annotation(ann_dir)

    if labels_list is None:
        labels_list = []
    if len(labels_list) == 0:
        add_labels(labels_list, annot_labels)
    else:
        check_labels(labels_list, annot_labels)

    test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
    pipeline = [dict(type="LoadAnnotations")]

    dataset = CustomDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=pipeline,
        classes=get_extended_label_names(labels_list),
        test_mode=test_mode,
    )
    dataset.test_mode = False

    dataset_items = []
    for item in dataset:
        annotations = create_annotation_from_hard_seg_map(hard_seg_map=item["gt_semantic_seg"], labels=labels_list)
        filename = os.path.join(item["img_prefix"], item["img_info"]["filename"])
        image = Image(file_path=filename)
        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=annotations)
        dataset_items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=subset))

    return dataset_items
