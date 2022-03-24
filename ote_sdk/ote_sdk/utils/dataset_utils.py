"""
Dataset utils
"""

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

from typing import List, Optional, Tuple

from ote_sdk.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.shapes.rectangle import Rectangle


def get_fully_annotated_idx(dataset: DatasetEntity) -> List[int]:
    """
    Find the indices of the fully annotated items in a dataset.
    A dataset item is fully annotated if local annotations are available, or if the item has the `normal` label.

    Args:
        dataset (DatasetEntity): Dataset that may contain both partially and fully annotated items

    Returns:
        List[int]: List of indices of the fully annotated dataset items.
    """
    local_idx = []
    for idx, gt_item in enumerate(dataset):
        local_annotations = [
            annotation
            for annotation in gt_item.get_annotations()
            if not Rectangle.is_full_box(annotation.shape)
        ]
        if (
            not any(label.is_anomalous for label in gt_item.get_shapes_labels())
            or len(local_annotations) > 0
        ):
            local_idx.append(idx)
    return local_idx


def get_local_subset(
    dataset: DatasetEntity, fully_annotated_idx: Optional[List[int]] = None
) -> DatasetEntity:
    """
    Extract a subset that contains only those dataset items that have local annotations.

    Args:
        dataset (DatasetEntity): Dataset from which we want to extract the locally annotated subset.
        fully_annotated_idx (Optional[List[int]]): The indices of the fully annotated dataset items. If not provided,
            the function will compute the indices before creating the subset.

    Returns:
        DatasetEntity: Output dataset with only local annotations
    """
    local_items = []
    if fully_annotated_idx is None:
        fully_annotated_idx = get_fully_annotated_idx(dataset)
    for idx in fully_annotated_idx:
        item = dataset[idx]

        local_annotations = [
            annotation
            for annotation in item.get_annotations()
            if not Rectangle.is_full_box(annotation.shape)
        ]
        # annotations with the normal label are considered local
        normal_annotations = [
            annotation
            for annotation in item.get_annotations()
            if not any(label.label.is_anomalous for label in annotation.get_labels())
        ]

        local_items.append(
            DatasetItemEntity(
                media=item.media,
                annotation_scene=AnnotationSceneEntity(
                    normal_annotations + local_annotations,
                    kind=AnnotationSceneKind.ANNOTATION,
                ),
                metadata=item.metadata,
                subset=item.subset,
                roi=item.roi,
                ignored_labels=item.ignored_labels,
            )
        )
    return DatasetEntity(local_items, purpose=dataset.purpose)


def get_global_subset(dataset: DatasetEntity) -> DatasetEntity:
    """
    Extract a subset that contains only the global annotations.

    Args:
        dataset (DatasetEntity): Dataset from which we want to extract the globally annotated subset.

    Returns:
        DatasetEntity: Output dataset with only global annotations
    """
    global_items = []
    for item in dataset:
        global_annotations = [
            annotation
            for annotation in item.get_annotations()
            if Rectangle.is_full_box(annotation.shape)
        ]
        global_items.append(
            DatasetItemEntity(
                media=item.media,
                annotation_scene=AnnotationSceneEntity(
                    global_annotations, kind=AnnotationSceneKind.ANNOTATION
                ),
                metadata=item.metadata,
                subset=item.subset,
                roi=item.roi,
                ignored_labels=item.ignored_labels,
            )
        )
    return DatasetEntity(global_items, purpose=dataset.purpose)


def split_local_global_dataset(
    dataset: DatasetEntity,
) -> Tuple[DatasetEntity, DatasetEntity]:
    """
    Split a dataset into the globally and locally annotated subsets.
    Args:
        dataset (DatasetEntity): Input dataset

    Returns:
        DatasetEntity: Globally annotated subset
        DatasetEntity: Locally annotated subset
    """
    global_dataset = get_global_subset(dataset)
    local_dataset = get_local_subset(dataset)
    return global_dataset, local_dataset


def split_local_global_resultset(
    resultset: ResultSetEntity,
) -> Tuple[ResultSetEntity, ResultSetEntity]:
    """
    Split a resultset into the globally and locally annotated resultsets.
    Args:
        resultset (ResultSetEntity): Input result set

    Returns:
        ResultSetEntity: Globally annotated result set
        ResultSetEntity: Locally annotated result set
    """
    global_gt_dataset, local_gt_dataset = split_local_global_dataset(
        resultset.ground_truth_dataset
    )
    local_idx = get_fully_annotated_idx(resultset.ground_truth_dataset)
    global_pred_dataset = get_global_subset(resultset.prediction_dataset)
    local_pred_dataset = get_local_subset(resultset.prediction_dataset, local_idx)

    global_resultset = ResultSetEntity(
        model=resultset.model,
        ground_truth_dataset=global_gt_dataset,
        prediction_dataset=global_pred_dataset,
        purpose=resultset.purpose,
    )
    local_resultset = ResultSetEntity(
        model=resultset.model,
        ground_truth_dataset=local_gt_dataset,
        prediction_dataset=local_pred_dataset,
        purpose=resultset.purpose,
    )
    return global_resultset, local_resultset


def contains_anomalous_images(dataset: DatasetEntity) -> bool:
    """Find the number of local annotations in a resultset."""
    for item in dataset:
        labels = item.get_shapes_labels()
        if any(label.is_anomalous for label in labels):
            return True
    return False
