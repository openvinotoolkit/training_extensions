"""
Dataset utils for OTE Anomaly
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

from typing import Tuple

from ote_sdk.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.shapes.rectangle import Rectangle


def filter_full_annotations(dataset) -> DatasetEntity:
    """
    Filter out the fully annotated images in the dataset.
    """
    fully_annotated = []
    for dataset_item in dataset:
        annotations = dataset_item.get_annotations()
        local_annotations = [annotation for annotation in annotations if not Rectangle.is_full_box(annotation.shape)]
        if not any(label.is_anomalous for label in dataset_item.get_shapes_labels()):
            fully_annotated.append(dataset_item)
        if len(local_annotations) > 0:
            fully_annotated.append(
                DatasetItemEntity(
                    media=dataset_item.media,
                    annotation_scene=AnnotationSceneEntity(local_annotations, kind=AnnotationSceneKind.ANNOTATION),
                    metadata=dataset_item.metadata,
                    subset=dataset_item.subset,
                    ignored_labels=dataset_item.ignored_labels,
                )
            )
    return DatasetEntity(fully_annotated, purpose=dataset.purpose)


def split_local_global_annotations(resultset) -> Tuple[ResultSetEntity, ResultSetEntity]:
    """Split resultset based on the type of available annotations."""
    # splits the dataset
    globally_annotated = []
    locally_annotated = []
    globally_predicted = []
    locally_predicted = []
    for gt_item, pred_item in zip(resultset.ground_truth_dataset, resultset.prediction_dataset):

        annotations = gt_item.get_annotations()
        global_annotations = [annotation for annotation in annotations if Rectangle.is_full_box(annotation.shape)]
        local_annotations = [annotation for annotation in annotations if not Rectangle.is_full_box(annotation.shape)]

        predictions = gt_item.get_annotations()
        global_predictions = [predictions for predictions in predictions if Rectangle.is_full_box(predictions.shape)]
        local_predictions = [predictions for predictions in predictions if not Rectangle.is_full_box(predictions.shape)]

        if not any(label.is_anomalous for label in gt_item.get_shapes_labels()):
            # normal images get added to both datasets
            globally_annotated.append(gt_item)
            locally_annotated.append(gt_item)
            globally_predicted.append(
                DatasetItemEntity(
                    media=pred_item.media,
                    annotation_scene=AnnotationSceneEntity(global_predictions, kind=AnnotationSceneKind.PREDICTION),
                    metadata=pred_item.metadata,
                    subset=pred_item.subset,
                    ignored_labels=pred_item.ignored_labels,
                )
            )
            locally_predicted.append(
                DatasetItemEntity(
                    media=pred_item.media,
                    annotation_scene=AnnotationSceneEntity(local_predictions, kind=AnnotationSceneKind.PREDICTION),
                    metadata=pred_item.metadata,
                    subset=pred_item.subset,
                    ignored_labels=pred_item.ignored_labels,
                )
            )
        else:  # image is abnormal
            globally_annotated.append(
                DatasetItemEntity(
                    media=gt_item.media,
                    annotation_scene=AnnotationSceneEntity(global_annotations, kind=AnnotationSceneKind.ANNOTATION),
                    metadata=gt_item.metadata,
                    subset=gt_item.subset,
                    ignored_labels=gt_item.ignored_labels,
                )
            )
            globally_predicted.append(
                DatasetItemEntity(
                    media=pred_item.media,
                    annotation_scene=AnnotationSceneEntity(global_predictions, kind=AnnotationSceneKind.PREDICTION),
                    metadata=pred_item.metadata,
                    subset=pred_item.subset,
                    ignored_labels=pred_item.ignored_labels,
                )
            )
            # add locally annotated dataset items
            if len(local_annotations) > 0:
                locally_annotated.append(
                    DatasetItemEntity(
                        media=gt_item.media,
                        annotation_scene=AnnotationSceneEntity(local_annotations, kind=AnnotationSceneKind.ANNOTATION),
                        metadata=gt_item.metadata,
                        subset=gt_item.subset,
                        ignored_labels=gt_item.ignored_labels,
                    )
                )
                locally_predicted.append(
                    DatasetItemEntity(
                        media=pred_item.media,
                        annotation_scene=AnnotationSceneEntity(local_predictions, kind=AnnotationSceneKind.PREDICTION),
                        metadata=pred_item.metadata,
                        subset=pred_item.subset,
                        ignored_labels=pred_item.ignored_labels,
                    )
                )
    global_gt_dataset = DatasetEntity(globally_annotated, purpose=resultset.ground_truth_dataset.purpose)
    local_gt_dataset = DatasetEntity(locally_annotated, purpose=resultset.ground_truth_dataset.purpose)
    global_pred_dataset = DatasetEntity(globally_predicted, purpose=resultset.prediction_dataset.purpose)
    local_pred_dataset = DatasetEntity(locally_predicted, purpose=resultset.prediction_dataset.purpose)

    global_resultset = ResultSetEntity(resultset.model, global_gt_dataset, global_pred_dataset, resultset.purpose)
    local_resultset = ResultSetEntity(resultset.model, local_gt_dataset, local_pred_dataset, resultset.purpose)

    return global_resultset, local_resultset


def contains_anomalous_images(resultset: ResultSetEntity) -> bool:
    """Find the number of local annotations in a resultset."""
    gt_dataset = resultset.ground_truth_dataset
    for item in gt_dataset:
        labels = item.get_shapes_labels()
        if any(label.is_anomalous for label in labels):
            return True
    return False
