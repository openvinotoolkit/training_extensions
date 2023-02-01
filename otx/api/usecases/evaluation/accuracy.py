"""This module contains the implementation of Accuracy performance provider."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import copy
import logging
from typing import List, Set, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelGroup
from otx.api.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    ColorPalette,
    MatrixChartInfo,
    MatrixMetric,
    MatrixMetricsGroup,
    MetricsGroup,
    Performance,
    ScoreMetric,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.evaluation.averaging import MetricAverageMethod
from otx.api.usecases.evaluation.basic_operations import (
    precision_per_class,
    recall_per_class,
)
from otx.api.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)

logger = logging.getLogger(__name__)


class Accuracy(IPerformanceProvider):
    """This class is responsible for providing Accuracy measures; mainly for Classification problems.

    The calculation both supports multi label and binary label predictions.

    Accuracy is the proportion of the predicted correct labels, to the total number (predicted and actual)
    labels for that instance. Overall accuracy is the average across all instances.

    Args:
        resultset (ResultSetEntity): ResultSet that score will be computed for
        average (MetricAverageMethod, optional): The averaging method, either MICRO or MACRO
            MICRO: compute average over all predictions in all label groups
            MACRO: compute accuracy per label group, return the average of the per-label-group accuracy scores
    """

    def __init__(
        self,
        resultset: ResultSetEntity,
        average: MetricAverageMethod = MetricAverageMethod.MICRO,
    ):
        self._unnormalized_matrices: List[MatrixMetric] = compute_unnormalized_confusion_matrices_from_resultset(
            resultset
        )

        # accuracy computation
        mean_accuracy = self._compute_accuracy(average=average, confusion_matrices=self._unnormalized_matrices)
        self._accuracy = ScoreMetric(value=mean_accuracy, name="Accuracy")

    @property
    def accuracy(self) -> ScoreMetric:
        """Returns the accuracy as ScoreMetric."""
        return self._accuracy

    def get_performance(self) -> Performance:
        """Returns the performance with accuracy and confusion metrics."""
        confusion_matrix_dashboard_metrics: List[MetricsGroup] = []

        # Use normalized matrix for UI
        normalized_matrices: List[MatrixMetric] = copy.deepcopy(self._unnormalized_matrices)
        for unnormalized_matrix in normalized_matrices:
            unnormalized_matrix.normalize()

        confusion_matrix_info = MatrixChartInfo(
            name="Confusion matrix",
            header="confusion",
            row_header="Predicted label",
            column_header="True label",
        )
        confusion_matrix_dashboard_metrics.append(
            MatrixMetricsGroup(metrics=normalized_matrices, visualization_info=confusion_matrix_info)
        )
        #  Compute precision and recall MetricGroups and append them to the dashboard metrics
        for _confusion_matrix in self._unnormalized_matrices:
            confusion_matrix_dashboard_metrics.append(precision_metrics_group(_confusion_matrix))
            confusion_matrix_dashboard_metrics.append(recall_metrics_group(_confusion_matrix))

        return Performance(score=self.accuracy, dashboard_metrics=confusion_matrix_dashboard_metrics)

    @staticmethod
    def _compute_accuracy(average: MetricAverageMethod, confusion_matrices: List[MatrixMetric]) -> float:
        """Compute accuracy using the confusion matrices.

        Args:
            average (MatricAverageMethod): The averaging method, either MICRO or MACRO
                MICRO: compute average over all predictions in all label groups
                MACRO: compute accuracy per label group, return the average of the per-label-group accuracy scores
            confusion_matrices (List[MatrixMetric]): the confusion matrices to compute accuracy from.
                MUST be unnormalized.

        Raises
            ValueError: when the ground truth dataset does not contain annotations
            RuntimeError: when the averaging methods is not known
        Returns:
            float: the accuracy score for the provided confusion matrix
        """
        # count correct predictions and total annotations
        correct_per_label_group = [np.trace(mat.matrix_values) for mat in confusion_matrices]
        total_per_label_group = [np.sum(mat.matrix_values) for mat in confusion_matrices]
        # check if all label groups have annotations
        if not np.any(total_per_label_group):
            raise ValueError("The ground truth dataset must contain annotations.")
        # return micro or macro average
        if average == MetricAverageMethod.MACRO:
            # compute accuracy for each label group, then average across groups, ignoring groups without annotations
            return np.nanmean(np.divide(correct_per_label_group, total_per_label_group))
        if average == MetricAverageMethod.MICRO:
            # average over all predictions in all label groups
            return np.sum(correct_per_label_group) / np.sum(total_per_label_group)

        raise RuntimeError(f"Unknown averaging method: {average}")


def precision_metrics_group(confusion_matrix: MatrixMetric) -> MetricsGroup:
    """Computes the precision per class based on a confusion matrix and returns them as ScoreMetrics in a MetricsGroup.

    Args:
        confusion_matrix: matrix to compute the precision per class for

    Returns:
        a BarMetricsGroup with the per class precision.
    """
    labels = confusion_matrix.row_labels
    if labels is None:
        # If no labels are given, just number the classes by index
        if confusion_matrix.matrix_values is not None:
            label_range = confusion_matrix.matrix_values.shape[0]
        else:
            label_range = 0
        labels = np.arange(label_range)

    per_class_precision = [
        ScoreMetric(class_, value=precision)
        for (class_, precision) in zip(labels, precision_per_class(confusion_matrix.matrix_values))
    ]

    return BarMetricsGroup(
        metrics=per_class_precision,
        visualization_info=BarChartInfo(
            name="Precision per class",
            palette=ColorPalette.LABEL,
        ),
    )


def recall_metrics_group(confusion_matrix: MatrixMetric) -> MetricsGroup:
    """Computes the recall per class based on a confusion matrix and returns them as ScoreMetrics in a MetricsGroup.

    Args:
        confusion_matrix: matrix to compute the recall per class for

    Returns:
        a BarMetricsGroup with the per class recall
    """
    labels = confusion_matrix.row_labels
    if labels is None:
        # If no labels are given, just number the classes by index
        if confusion_matrix.matrix_values is not None:
            label_range = confusion_matrix.matrix_values.shape[0]
        else:
            label_range = 0
        labels = np.arange(label_range)

    per_class_recall = [
        ScoreMetric(class_, value=recall)
        for (class_, recall) in zip(labels, recall_per_class(confusion_matrix.matrix_values))
    ]

    return BarMetricsGroup(
        metrics=per_class_recall,
        visualization_info=BarChartInfo(
            name="Recall per class",
            palette=ColorPalette.LABEL,
        ),
    )


def __get_gt_and_predicted_label_indices_from_resultset(
    resultset: ResultSetEntity,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    """Returns the label indices lists for ground truth and prediction datasets in a tuple.

    Args:
        resultset

    Returns:
        a tuple containing two lists. The first list contains the ground truth label indices, and the second contains
        the prediction label indices.
    """
    true_label_idx = []
    predicted_label_idx = []

    gt_dataset: DatasetEntity = resultset.ground_truth_dataset
    pred_dataset: DatasetEntity = resultset.prediction_dataset

    gt_dataset.sort_items()
    pred_dataset.sort_items()

    # Iterate over each dataset item, and collect the labels for this item (pred and gt)
    task_labels = resultset.model.configuration.get_label_schema().get_labels(include_empty=True)
    for gt_item, pred_item in zip(gt_dataset, pred_dataset):
        if isinstance(gt_item, DatasetItemEntity) and isinstance(pred_item, DatasetItemEntity):
            true_label_idx.append({task_labels.index(label) for label in gt_item.get_roi_labels(task_labels)})
            predicted_label_idx.append({task_labels.index(label) for label in pred_item.get_roi_labels(task_labels)})

    return true_label_idx, predicted_label_idx


def __compute_unnormalized_confusion_matrices_for_label_group(
    true_label_idx: List[Set[int]],
    predicted_label_idx: List[Set[int]],
    label_group: LabelGroup,
    task_labels: List[LabelEntity],
) -> MatrixMetric:
    """Returns matrix metric for a certain label group.

    Args:
        true_label_idx (List[Set[int]]): list of sets of label indices for the ground truth dataset
        predicted_label_idx (List[Set[int]]): list of sets of label indices for the prediction dataset
        label_group (LabelGroup): label group to compute the confusion matrix for
        task_labels (List[LabelEntity]): list of labels for the task

    Returns:
        MatrixMetric: confusion matrix for the label group
    """
    map_task_labels_idx_to_group_idx = {
        task_labels.index(label): i_group for i_group, label in enumerate(label_group.labels)
    }
    set_group_labels_idx = set(map_task_labels_idx_to_group_idx.keys())
    group_label_names = [task_labels[label_idx].name for label_idx in set_group_labels_idx]

    if len(group_label_names) == 1:
        # Single-class
        # we use "not" to make presence of a class to be at index 0, while the absence of it at index 1
        y_true = [int(not set_group_labels_idx.issubset(true_labels)) for true_labels in true_label_idx]
        y_pred = [int(not set_group_labels_idx.issubset(pred_labels)) for pred_labels in predicted_label_idx]
        group_label_names += [f"~ {group_label_names[0]}"]
        column_labels = group_label_names.copy()
        remove_last_row = False
    else:
        # Multiclass
        undefined_idx = len(group_label_names)  # to define missing value

        # find the intersections between GT and task labels, and Prediction and task labels
        true_intersections = [true_labels.intersection(set_group_labels_idx) for true_labels in true_label_idx]
        pred_intersections = [pred_labels.intersection(set_group_labels_idx) for pred_labels in predicted_label_idx]

        # map the intersection to 0-index value
        y_true = [
            map_task_labels_idx_to_group_idx[list(true_intersection)[0]]
            if len(true_intersection) != 0
            else undefined_idx
            for true_intersection in true_intersections
        ]
        y_pred = [
            map_task_labels_idx_to_group_idx[list(pred_intersection)[0]]
            if len(pred_intersection) != 0
            else undefined_idx
            for pred_intersection in pred_intersections
        ]

        column_labels = group_label_names.copy()
        column_labels.append("Other")
        remove_last_row = True

    matrix_data = sklearn_confusion_matrix(y_true, y_pred, labels=list(range(len(column_labels))))
    if remove_last_row:
        # matrix clean up
        matrix_data = np.delete(matrix_data, -1, 0)
        if sum(matrix_data[:, -1]) == 0:
            # if none of the GT is classified as classes from other groups, clean it up too
            matrix_data = np.delete(matrix_data, -1, 1)
            column_labels.remove(column_labels[-1])

    # Use unnormalized matrix for statistics computation (accuracy, precision, recall)
    return MatrixMetric(
        name=f"{label_group.name}",
        matrix_values=matrix_data,
        row_labels=group_label_names,
        column_labels=column_labels,
        normalize=False,
    )


def compute_unnormalized_confusion_matrices_from_resultset(
    resultset: ResultSetEntity,
) -> List[MatrixMetric]:
    """Computes an (unnormalized) confusion matrix for every label group in the resultset.

    Args:
        resultset: the input resultset

    Returns:
        the computed unnormalized confusion matrices
    """

    if len(resultset.ground_truth_dataset) == 0 or len(resultset.prediction_dataset) == 0:
        raise ValueError("Cannot compute the confusion matrix of an empty result set.")

    unnormalized_confusion_matrices: List[MatrixMetric] = []
    (
        true_label_idx,
        predicted_label_idx,
    ) = __get_gt_and_predicted_label_indices_from_resultset(resultset)
    task_labels = resultset.model.configuration.get_label_schema().get_labels(include_empty=False)

    # Confusion matrix computation
    for label_group in resultset.model.configuration.get_label_schema().get_groups():
        matrix = __compute_unnormalized_confusion_matrices_for_label_group(
            true_label_idx, predicted_label_idx, label_group, task_labels
        )
        unnormalized_confusion_matrices.append(matrix)

    return unnormalized_confusion_matrices
