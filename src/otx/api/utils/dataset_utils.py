"""Dataset utils."""

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

from typing import List, Optional, Tuple, Union

import numpy as np

from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.utils.vis_utils import get_actmap


def get_fully_annotated_idx(dataset: DatasetEntity) -> List[int]:
    """Find the indices of the fully annotated items in a dataset.

    A dataset item is fully annotated if local annotations are available, or if the item has the `normal` label.

    Args:
        dataset (DatasetEntity): Dataset that may contain both partially and fully annotated items.

    Returns:
        List[int]: List of indices of the fully annotated dataset items.
    """
    local_idx = []
    for idx, gt_item in enumerate(dataset):
        local_annotations = [
            annotation for annotation in gt_item.get_annotations() if not Rectangle.is_full_box(annotation.shape)
        ]
        if not any(label.is_anomalous for label in gt_item.get_shapes_labels()) or len(local_annotations) > 0:
            local_idx.append(idx)
    return local_idx


def get_local_subset(
    dataset: DatasetEntity,
    fully_annotated_idx: Optional[List[int]] = None,
    include_normal: bool = True,
) -> DatasetEntity:
    """Extract a subset that contains only those dataset items that have local annotations.

    Args:
        dataset (DatasetEntity): Dataset from which we want to extract the locally annotated subset.
        fully_annotated_idx (Optional[List[int]]): The indices of the fully annotated dataset items. If not provided,
            the function will compute the indices before creating the subset.
        include_normal (bool): When true, global normal annotations will be included in the local dataset.

    Returns:
        DatasetEntity: Output dataset with only local annotations
    """
    local_items = []
    if fully_annotated_idx is None:
        fully_annotated_idx = get_fully_annotated_idx(dataset)
    for idx in fully_annotated_idx:
        item = dataset[idx]

        local_annotations = [
            annotation for annotation in item.get_annotations() if not Rectangle.is_full_box(annotation.shape)
        ]
        # annotations with the normal label are considered local
        if include_normal:
            local_annotations.extend(
                [
                    annotation
                    for annotation in item.get_annotations()
                    if not any(label.label.is_anomalous for label in annotation.get_labels())
                ]
            )
        local_items.append(
            DatasetItemEntity(
                media=item.media,
                annotation_scene=AnnotationSceneEntity(
                    local_annotations,
                    kind=item.annotation_scene.kind,
                ),
                metadata=item.get_metadata(),
                subset=item.subset,
                roi=item.roi,
                ignored_labels=item.ignored_labels,
            )
        )
    return DatasetEntity(local_items, purpose=dataset.purpose)


def get_global_subset(dataset: DatasetEntity) -> DatasetEntity:
    """Extract a subset that contains only the global annotations.

    Args:
        dataset (DatasetEntity): Dataset from which we want to extract the globally annotated subset.

    Returns:
        DatasetEntity: Output dataset with only global annotations
    """
    global_items = []
    for item in dataset:
        global_annotations = [
            annotation for annotation in item.get_annotations() if Rectangle.is_full_box(annotation.shape)
        ]
        global_items.append(
            DatasetItemEntity(
                media=item.media,
                annotation_scene=AnnotationSceneEntity(global_annotations, kind=item.annotation_scene.kind),
                metadata=item.get_metadata(),
                subset=item.subset,
                roi=item.roi,
                ignored_labels=item.ignored_labels,
            )
        )
    return DatasetEntity(global_items, purpose=dataset.purpose)


def split_local_global_dataset(
    dataset: DatasetEntity,
) -> Tuple[DatasetEntity, DatasetEntity]:
    """Split a dataset into the globally and locally annotated subsets.

    Args:
        dataset (DatasetEntity): Input dataset

    Returns:
        Tuple[DatasetEntity, DatasetEntity]: Tuple of the globally and locally annotated subsets.
    """
    global_dataset = get_global_subset(dataset)
    local_dataset = get_local_subset(dataset)
    return global_dataset, local_dataset


def split_local_global_resultset(
    resultset: ResultSetEntity,
) -> Tuple[ResultSetEntity, ResultSetEntity]:
    """Split a resultset into the globally and locally annotated resultsets.

    Args:
        resultset (ResultSetEntity): Input resultset

    Returns:
        Tuple[ResultSetEntity, ResultSetEntity]: Tuple of the globally and locally annotated resultsets.
    """
    global_gt_dataset = get_global_subset(resultset.ground_truth_dataset)
    local_gt_dataset = get_local_subset(resultset.ground_truth_dataset, include_normal=False)
    local_idx = get_fully_annotated_idx(resultset.ground_truth_dataset)
    global_pred_dataset = get_global_subset(resultset.prediction_dataset)
    local_pred_dataset = get_local_subset(resultset.prediction_dataset, local_idx, include_normal=False)

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
    """Check if a dataset contains any items with the anomalous label.

    Args:
        dataset (DatasetEntity): Dataset to check for anomalous items.

    Returns:
        bool: True if the dataset contains anomalous items, False otherwise.
    """
    for item in dataset:
        labels = item.get_shapes_labels()
        if any(label.is_anomalous for label in labels):
            return True
    return False


# pylint: disable-msg=too-many-locals
def add_saliency_maps_to_dataset_item(
    dataset_item: DatasetItemEntity,
    saliency_map: Union[List[Optional[np.ndarray]], np.ndarray],
    model: Optional[ModelEntity],
    labels: List[LabelEntity],
    predicted_scored_labels: Optional[List[ScoredLabel]] = None,
    explain_predicted_classes: bool = True,
    process_saliency_maps: bool = False,
):
    """Add saliency maps (2D array for class-agnostic saliency map,
    3D array or list or 2D arrays for class-wise saliency maps) to a single dataset item."""
    if isinstance(saliency_map, list):
        class_wise_saliency_map = True
    elif isinstance(saliency_map, np.ndarray):
        if saliency_map.ndim == 2:
            class_wise_saliency_map = False
        elif saliency_map.ndim == 3:
            class_wise_saliency_map = True
        else:
            raise ValueError(f"Saliency map has to be 2 or 3-dimensional array, " f"but got {saliency_map.ndim} dims.")
    else:
        raise TypeError("Check saliency_map, it has to be list or np.ndarray.")

    if class_wise_saliency_map:
        # Multiple saliency maps per image (class-wise saliency map), support e.g. ReciproCAM
        if explain_predicted_classes:
            # Explain only predicted classes
            if predicted_scored_labels is None:
                raise ValueError("To explain only predictions, list of predicted scored labels have to be provided.")

            explain_targets = set()
            for scored_label in predicted_scored_labels:
                if scored_label.label is not None:  # Check for an empty label
                    explain_targets.add(scored_label.label)
        else:
            # Explain all classes
            explain_targets = set(labels)

        for class_id, class_wise_saliency_map in enumerate(saliency_map):
            label = labels[class_id]
            if class_wise_saliency_map is not None and label in explain_targets:
                if process_saliency_maps:
                    class_wise_saliency_map = get_actmap(
                        class_wise_saliency_map, (dataset_item.width, dataset_item.height)
                    )
                saliency_media = ResultMediaEntity(
                    name=label.name,
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=class_wise_saliency_map,
                    roi=dataset_item.roi,
                    label=label,
                )
                dataset_item.append_metadata_item(saliency_media, model=model)
    else:
        # Single saliency map per image, support e.g. ActivationMap
        if process_saliency_maps:
            saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
        saliency_media = ResultMediaEntity(
            name="Saliency Map",
            type="saliency_map",
            annotation_scene=dataset_item.annotation_scene,
            numpy=saliency_map,
            roi=dataset_item.roi,
        )
        dataset_item.append_metadata_item(saliency_media, model=model)


def non_linear_normalization(saliency_map: np.ndarray) -> np.ndarray:
    """Use non-linear normalization y=x**1.5 for 2D saliency maps."""

    min_soft_score = np.min(saliency_map)
    # make merged_map distribution positive to perform non-linear normalization y=x**1.5
    saliency_map = (saliency_map - min_soft_score) ** 1.5

    max_soft_score = np.max(saliency_map)
    saliency_map = 255.0 / (max_soft_score + 1e-12) * saliency_map

    return np.uint8(np.floor(saliency_map))
