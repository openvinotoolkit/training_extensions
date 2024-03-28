"""This module contains the f-measure performance provider class."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    ColorPalette,
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    MultiScorePerformance,
    ScoreMetric,
    TextChartInfo,
    TextMetricsGroup,
    VisualizationType,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from otx.api.utils.shape_factory import ShapeFactory

logger = logging.getLogger(__name__)
ALL_CLASSES_NAME = "All Classes"


def intersection_box(
    box1: Tuple[float, float, float, float, str, float], box2: Tuple[float, float, float, float, str, float]
) -> Tuple[float, float, float, float]:
    """Calculate the intersection box of two bounding boxes.

    Args:
        box1 (Tuple[float, float, float, float, str, float]): (x1, y1, x2, y2, class, score)
        box2 (Tuple[float, float, float, float, str, float]): (x1, y1, x2, y2, class, score)

    Returns:
        Tuple[float, float, float, float]: (x_left, x_right, y_bottom, y_top)
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    return (x_left, x_right, y_bottom, y_top)


def bounding_box_intersection_over_union(
    box1: Tuple[float, float, float, float, str, float], box2: Tuple[float, float, float, float, str, float]
) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (Tuple[float, float, float, float, str, float]): (x1, y1, x2, y2, class, score)
        box2 (Tuple[float, float, float, float, str, float]): (x1, y1, x2, y2, class, score)

    Raises:
        ValueError: In case the IoU is outside of [0.0, 1.0]

    Returns:
        float: Intersection-over-union of box1 and box2.
    """

    x_left, x_right, y_bottom, y_top = intersection_box(box1, box2)

    if x_right <= x_left or y_bottom <= y_top:
        iou = 0.0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        bb2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(bb1_area + bb2_area - intersection_area)
        if union_area == 0:
            iou = 0.0
        else:
            iou = intersection_area / union_area
    if iou < 0.0 or iou > 1.0:
        raise ValueError(f"intersection over union should be in range [0,1], actual={iou}")
    return iou


def get_iou_matrix(
    ground_truth: List[Tuple[float, float, float, float, str, float]],
    predicted: List[Tuple[float, float, float, float, str, float]],
) -> np.ndarray:
    """Constructs an iou matrix of shape [num_ground_truth_boxes, num_predicted_boxes].

    Each cell(x,y) in the iou matrix contains the intersection over union of ground truth box(x) and predicted box(y)
    An iou matrix corresponds to a single image

    Args:
        ground_truth (List[Tuple[float, float, float, float, str, float]]): List of ground truth boxes.
            Each box is a list of (x,y) coordinates and a label.
            a box: [x1: float, y1, x2, y2, class: str, score: float]
            boxes_per_image: [box1, box2, …]
            boxes1: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]
        predicted (List[Tuple[float, float, float, float, str, float]]): List of predicted boxes.
            Each box is a list of (x,y) coordinates and a label.
            a box: [x1: float, y1, x2, y2, class: str, score: float]
            boxes_per_image: [box1, box2, …]
            boxes2: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]

    Returns:
        np.ndarray: IoU matrix of shape [ground_truth_boxes, predicted_boxes]
    """
    matrix = np.array(
        [[bounding_box_intersection_over_union(gts, preds) for preds in predicted] for gts in ground_truth]
    )
    return matrix


def get_n_false_negatives(iou_matrix: np.ndarray, iou_threshold: float) -> int:
    """Get the number of false negatives inside the IoU matrix for a given threshold.

    The first loop accounts for all the ground truth boxes which do not have a high enough iou with any predicted
    box (they go undetected)
    The second loop accounts for the much rarer case where two ground truth boxes are detected by the same predicted
    box. The principle is that each ground truth box requires a unique prediction box

    Args:
        iou_matrix (np.ndarray): IoU matrix of shape [ground_truth_boxes, predicted_boxes]
        iou_threshold (float): IoU threshold to use for the false negatives.

    Returns:
        int: Number of false negatives
    """
    n_false_negatives = 0
    for row in iou_matrix:
        if max(row) < iou_threshold:
            n_false_negatives += 1
    for column in np.rot90(iou_matrix):
        indices = np.where(column > iou_threshold)
        n_false_negatives += max(len(indices[0]) - 1, 0)
    return n_false_negatives


class _Metrics:
    """This class collects the metrics related to detection.

    Args:
        f_measure (float): F-measure of the model.
        precision (float): Precision of the model.
        recall (float): Recall of the model.
    """

    def __init__(self, f_measure: float, precision: float, recall: float):
        self.f_measure = f_measure
        self.precision = precision
        self.recall = recall


class _ResultCounters:
    """This class collects the number of prediction, TP and FN.

    Args:
        n_false_negatives (int): Number of false negatives.
        n_true (int): Number of true positives.
        n_predictions (int): Number of predictions.
    """

    def __init__(self, n_false_negatives: int, n_true: int, n_predicted: int):
        self.n_false_negatives = n_false_negatives
        self.n_true = n_true
        self.n_predicted = n_predicted

    def calculate_f_measure(self) -> _Metrics:
        """Calculates and returns precision, recall, and f-measure.

        Returns:
            _Metrics: _Metrics object with Precision, recall, and f-measure.
        """
        n_true_positives = self.n_true - self.n_false_negatives

        if self.n_predicted == 0:
            precision = 1.0
            recall = 0.0
        elif self.n_true == 0:
            precision = 0.0
            recall = 1.0
        else:
            precision = n_true_positives / self.n_predicted
            recall = n_true_positives / self.n_true

        f_measure = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
        return _Metrics(f_measure, precision, recall)


class _AggregatedResults:
    """This class collects the aggregated results for F-measure.

    The result contains:
        - f_measure_curve
        - precision_curve
        - recall_curve
        - all_classes_f_measure_curve
        - best_f_measure
        - best_threshold
        - best_f_measure_metrics

    Args:
        classes (List[str]): List of classes.
    """

    def __init__(self, classes: List[str]):
        self.f_measure_curve: Dict[str, List[float]] = {class_name: [] for class_name in classes}
        self.precision_curve: Dict[str, List[float]] = {class_name: [] for class_name in classes}
        self.recall_curve: Dict[str, List[float]] = {class_name: [] for class_name in classes}
        self.all_classes_f_measure_curve: List[float] = []
        self.best_f_measure: float = 0.0
        self.best_threshold: float = 0.0
        self.best_f_measure_metrics: _Metrics = _Metrics(0.0, 0.0, 0.0)


class _OverallResults:
    """This class collects the overall results that is computed by the F-measure performance provider.

    Args:
        per_confidence (_AggregatedResults): _AggregatedResults object for each confidence level.
        per_nms (Optional[_AggregatedResults]): _AggregatedResults object for each NMS threshold.
        best_f_measure_per_class (Dict[str, float]): Best f-measure per class.
        best_f_measure (float): Best f-measure.
    """

    def __init__(
        self,
        per_confidence: _AggregatedResults,
        per_nms: Optional[_AggregatedResults],
        best_f_measure_per_class: Dict[str, float],
        best_f_measure: float,
    ):
        self.per_confidence = per_confidence
        self.per_nms = per_nms
        self.best_f_measure_per_class = best_f_measure_per_class
        self.best_f_measure = best_f_measure


class _FMeasureCalculator:
    """This class contains the functions to calculate FMeasure.

    Args:
        ground_truth_boxes_per_image (List[List[Tuple[float, float, float, float, str, float]]]):
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
                ground_truth_boxes_per_image: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]
        prediction_boxes_per_image (List[List[Tuple[float, float, float, float, str, float]]]):
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
                predicted_boxes_per_image: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]
    """

    def __init__(
        self,
        ground_truth_boxes_per_image: List[List[Tuple[float, float, float, float, str, float]]],
        prediction_boxes_per_image: List[List[Tuple[float, float, float, float, str, float]]],
    ):
        self.ground_truth_boxes_per_image = ground_truth_boxes_per_image
        self.prediction_boxes_per_image = prediction_boxes_per_image
        self.confidence_range = [0.025, 1.0, 0.025]
        self.nms_range = [0.1, 1, 0.05]
        self.default_confidence_threshold = 0.35

    def evaluate_detections(
        self,
        classes: List[str],
        iou_threshold: float = 0.5,
        result_based_nms_threshold: bool = False,
        cross_class_nms: bool = False,
    ) -> _OverallResults:
        """Evaluates detections by computing f_measures across multiple confidence thresholds and iou thresholds.

        By default, this function evaluates 39 confidence thresholds, finds the best confidence threshold and appends
        it to the result Dict
        Each one of the (default 39+20) pairs of confidence and nms thresholds is used to evaluate the f-measure for
        each class, then the intermediate metrics are summed across classes to compute an all_classes f_measure.
        Finally, the best results across all evaluations are appended to the result dictionary along with the thresholds
        used to achieve them.

        Args:
            classes (List[str]): Names of classes to be evaluated.
            iou_threshold (float): IOU threshold. Defaults to 0.5.
            result_based_nms_threshold (bool): Boolean that determines whether multiple nms threshold are examined.
                Defaults to False.
            cross_class_nms (bool): Set to True to perform NMS between boxes with different classes. Defaults to False.

        Returns:
            _OverallResults: _OverallResults object with the result statistics (e.g F-measure).
        """

        best_f_measure_per_class = {}

        results_per_confidence = self.get_results_per_confidence(
            classes=classes,
            confidence_range=self.confidence_range,
            iou_threshold=iou_threshold,
        )

        best_f_measure = results_per_confidence.best_f_measure

        for class_name in classes:
            best_f_measure_per_class[class_name] = max(results_per_confidence.f_measure_curve[class_name])

        results_per_nms: Optional[_AggregatedResults] = None

        if result_based_nms_threshold:
            results_per_nms = self.get_results_per_nms(
                classes=classes,
                iou_threshold=iou_threshold,
                min_f_measure=results_per_confidence.best_f_measure,
                cross_class_nms=cross_class_nms,
            )

            for class_name in classes:
                best_f_measure_per_class[class_name] = max(results_per_nms.f_measure_curve[class_name])

        result = _OverallResults(
            results_per_confidence,
            results_per_nms,
            best_f_measure_per_class,
            best_f_measure,
        )

        return result

    def get_results_per_confidence(
        self, classes: List[str], confidence_range: List[float], iou_threshold: float
    ) -> _AggregatedResults:
        """Returns the results for confidence threshold in range confidence_range.

        Varies confidence based on confidence_range, the results are appended in a dictionary and returned, it also
        returns the best f_measure found and the confidence threshold used to get said f_measure

        Args:
            classes (List[str]): Names of classes to be evaluated.
            confidence_range (List[float]): List of confidence thresholds to be evaluated.
            iou_threshold (float): IoU threshold to use for false negatives.

        Returns:
            _AggregatedResults: _AggregatedResults object with the result statistics (e.g F-measure).
        """
        result = _AggregatedResults(classes)
        result.best_threshold = 0.1

        for confidence_threshold in np.arange(*confidence_range):
            result_point = self.evaluate_classes(
                classes=classes.copy(),
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold,
            )
            all_classes_f_measure = result_point[ALL_CLASSES_NAME].f_measure
            result.all_classes_f_measure_curve.append(all_classes_f_measure)

            for class_name in classes:
                result.f_measure_curve[class_name].append(result_point[class_name].f_measure)
                result.precision_curve[class_name].append(result_point[class_name].precision)
                result.recall_curve[class_name].append(result_point[class_name].recall)
            if all_classes_f_measure > 0.0 and all_classes_f_measure >= result.best_f_measure:
                result.best_f_measure = all_classes_f_measure
                result.best_threshold = confidence_threshold
                result.best_f_measure_metrics = result_point[ALL_CLASSES_NAME]
        return result

    def get_results_per_nms(
        self,
        classes: List[str],
        iou_threshold: float,
        min_f_measure: float,
        cross_class_nms: bool = False,
    ) -> _AggregatedResults:
        """Returns results for nms threshold in range nms_range.

        First, we calculate the critical nms of each box, meaning the nms_threshold
        that would cause it to be disappear
        This is an expensive O(n**2) operation, however, doing this makes filtering for every single nms_threshold much
        faster at O(n)

        Args:
            classes (List[str]): List of classes
            iou_threshold (float): IoU threshold
            min_f_measure (float): the minimum F-measure required to select a NMS threshold
            cross_class_nms (bool): set to True to perform NMS between boxes with different classes. Defaults to False.

        Returns:
            _AggregatedResults: Object containing the results for each NMS threshold value
        """
        result = _AggregatedResults(classes)
        result.best_f_measure = min_f_measure
        result.best_threshold = 0.5

        critical_nms_per_image = self.__get_critical_nms(self.prediction_boxes_per_image, cross_class_nms)

        for nms_threshold in np.arange(*self.nms_range):
            predicted_boxes_per_image_per_nms = self.__filter_nms(
                self.prediction_boxes_per_image, critical_nms_per_image, nms_threshold
            )
            boxes_pair_for_nms = _FMeasureCalculator(
                self.ground_truth_boxes_per_image, predicted_boxes_per_image_per_nms
            )
            result_point = boxes_pair_for_nms.evaluate_classes(
                classes=classes.copy(),
                iou_threshold=iou_threshold,
                confidence_threshold=self.default_confidence_threshold,
            )
            all_classes_f_measure = result_point[ALL_CLASSES_NAME].f_measure
            result.all_classes_f_measure_curve.append(all_classes_f_measure)

            for class_name in classes:
                result.f_measure_curve[class_name].append(result_point[class_name].f_measure)
                result.precision_curve[class_name].append(result_point[class_name].precision)
                result.recall_curve[class_name].append(result_point[class_name].recall)

            if all_classes_f_measure > 0.0 and all_classes_f_measure >= result.best_f_measure:
                result.best_f_measure = all_classes_f_measure
                result.best_threshold = nms_threshold
                result.best_f_measure_metrics = result_point[ALL_CLASSES_NAME]
        return result

    def evaluate_classes(
        self, classes: List[str], iou_threshold: float, confidence_threshold: float
    ) -> Dict[str, _Metrics]:
        """Returns Dict of f_measure, precision and recall for each class.

        Args:
            classes (List[str]): List of classes to be evaluated.
            iou_threshold (float): IoU threshold to use for false negatives.
            confidence_threshold (float): Confidence threshold to use for false negatives.

        Returns:
            Dict[str, _Metrics]: The metrics (e.g. F-measure) for each class.
        """
        result: Dict[str, _Metrics] = {}

        all_classes_counters = _ResultCounters(0, 0, 0)

        if ALL_CLASSES_NAME in classes:
            classes.remove(ALL_CLASSES_NAME)
        for class_name in classes:
            metrics, counters = self.get_f_measure_for_class(
                class_name=class_name,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold,
            )
            result[class_name] = metrics
            all_classes_counters.n_false_negatives += counters.n_false_negatives
            all_classes_counters.n_true += counters.n_true
            all_classes_counters.n_predicted += counters.n_predicted

        # for all classes
        result[ALL_CLASSES_NAME] = all_classes_counters.calculate_f_measure()
        return result

    def get_f_measure_for_class(
        self, class_name: str, iou_threshold: float, confidence_threshold: float
    ) -> Tuple[_Metrics, _ResultCounters]:
        """Get f_measure for specific class, iou threshold, and confidence threshold.

        In order to reduce the number of redundant iterations and allow for cleaner, more general code later on,
        all boxes are filtered at this stage by class and predicted boxes are filtered by confidence threshold

        Args:

            class_name (str): Name of the class for which the F measure is computed
            iou_threshold (float): IoU threshold
            confidence_threshold (float): Confidence threshold

        Returns:
            Tuple[_Metrics, _ResultCounters]: a structure containing the statistics (e.g. f_measure) and a structure
            containing the intermediated counters used to derive the stats (e.g. num. false positives)
        """
        class_ground_truth_boxes_per_image = self.__filter_class(self.ground_truth_boxes_per_image, class_name)
        confidence_predicted_boxes_per_image = self.__filter_confidence(
            self.prediction_boxes_per_image, confidence_threshold
        )
        class_predicted_boxes_per_image = self.__filter_class(confidence_predicted_boxes_per_image, class_name)
        if len(class_ground_truth_boxes_per_image) > 0:
            boxes_pair_per_class = _FMeasureCalculator(
                ground_truth_boxes_per_image=class_ground_truth_boxes_per_image,
                prediction_boxes_per_image=class_predicted_boxes_per_image,
            )
            result_counters = boxes_pair_per_class.get_counters(iou_threshold=iou_threshold)
            result_metrics = result_counters.calculate_f_measure()
            results = (result_metrics, result_counters)
        else:
            logger.warning("No ground truth images supplied for f-measure calculation.")
            # [f_measure, precision, recall, n_false_negatives, n_true, n_predicted]
            results = (_Metrics(0.0, 0.0, 0.0), _ResultCounters(0, 0, 0))
        return results

    @staticmethod
    def __get_critical_nms(
        boxes_per_image: List[List[Tuple[float, float, float, float, str, float]]], cross_class_nms: bool = False
    ) -> List[List[float]]:
        """Return list of critical NMS values for each box in each image.

        Maps each predicted box to the highest nms-threshold which would suppress that box, aka the smallest
        nms_threshold before the box disappears.
        Having these values allows us to later filter by nms-threshold in O(n) rather than O(n**2)
        Highest losing iou, holds the value of the highest iou that a box has with any
        other box of the same class and higher confidence score.

        Args:
            boxes_per_image (List[List[Tuple[float, float, float, float, str, float]]]): List of predicted boxes per
                image.
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
            cross_class_nms (bool): Whether to use cross class NMS.

        Returns:
            List[List[float]]: List of critical NMS values for each box in each image.
        """
        critical_nms_per_image = []
        for boxes in boxes_per_image:
            critical_nms_per_box = []
            for box1 in boxes:
                highest_losing_iou = 0.0
                for box2 in boxes:
                    iou = bounding_box_intersection_over_union(box1, box2)
                    if (
                        (cross_class_nms or box1[FMeasure.box_class_index] == box2[FMeasure.box_class_index])
                        # TODO boxes Tuple should be refactored to dataclass.
                        and box1[FMeasure.box_score_index] < box2[FMeasure.box_score_index]  # type: ignore[operator]
                        and iou > highest_losing_iou
                    ):
                        highest_losing_iou = iou
                critical_nms_per_box.append(highest_losing_iou)
            critical_nms_per_image.append(critical_nms_per_box)
        return critical_nms_per_image

    @staticmethod
    def __filter_nms(
        boxes_per_image: List[List[Tuple[float, float, float, float, str, float]]],
        critical_nms: List[List[float]],
        nms_threshold: float,
    ) -> List[List[Tuple[float, float, float, float, str, float]]]:
        """Filters out predicted boxes whose critical nms is higher than the given nms_threshold.

        Args:
            boxes_per_image (List[List[Tuple[float, float, float, float, str, float]]]): List of boxes per image.
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
            critical_nms (List[List[float]]): List of list of critical nms for each box in each image
            nms_threshold (float): NMS threshold used for filtering

        Returns:
            List[List[Tuple[float, float, float, float, str, float]]]: List of list of filtered boxes in each image
        """
        new_boxes_per_image = []
        for boxes, boxes_nms in zip(boxes_per_image, critical_nms):
            new_boxes = []
            for box, nms in zip(boxes, boxes_nms):
                if nms < nms_threshold:
                    new_boxes.append(box)
            new_boxes_per_image.append(new_boxes)
        return new_boxes_per_image

    @staticmethod
    def __filter_class(
        boxes_per_image: List[List[Tuple[float, float, float, float, str, float]]], class_name: str
    ) -> List[List[Tuple[float, float, float, float, str, float]]]:
        """Filters boxes to only keep members of one class.

        Args:
            boxes_per_image (List[List[Tuple[float, float, float, float, str, float]]]): a list of lists of boxes
            class_name (str): Name of the class for which the boxes are filtered

        Returns:
            List[List[Tuple[float, float, float, float, str, float]]]: a list of lists of boxes
        """
        filtered_boxes_per_image = []
        for boxes in boxes_per_image:
            filtered_boxes = []
            for box in boxes:
                # TODO boxes Tuple should be refactored to dataclass. This way we can access box.class
                if box[FMeasure.box_class_index].lower() == class_name.lower():  # type: ignore[union-attr]
                    filtered_boxes.append(box)
            filtered_boxes_per_image.append(filtered_boxes)
        return filtered_boxes_per_image

    @staticmethod
    def __filter_confidence(
        boxes_per_image: List[List[Tuple[float, float, float, float, str, float]]], confidence_threshold: float
    ) -> List[List[Tuple[float, float, float, float, str, float]]]:
        """Filters boxes to only keep ones with higher confidence than a given confidence threshold.

        Args:
            boxes_per_image (List[List[Tuple[float, float, float, float, str, float]]]):
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
            confidence_threshold (float): Confidence threshold

        Returns:
            List[List[Tuple[float, float, float, float, str, float]]]: Boxes with higher confidence than the given
                threshold.
        """
        filtered_boxes_per_image = []
        for boxes in boxes_per_image:
            filtered_boxes = []
            for box in boxes:
                if float(box[FMeasure.box_score_index]) > confidence_threshold:
                    filtered_boxes.append(box)
            filtered_boxes_per_image.append(filtered_boxes)
        return filtered_boxes_per_image

    def get_counters(self, iou_threshold: float) -> _ResultCounters:
        """Return counts of true positives, false positives and false negatives for a given iou threshold.

        For each image (the loop), compute the number of false negatives, the number of predicted boxes, and the number
        of ground truth boxes, then add each value to its corresponding counter

        Args:
            iou_threshold (float): IoU threshold

        Returns:
            _ResultCounters: Structure containing the number of false negatives, true positives and predictions.
        """
        n_false_negatives = 0
        n_true = 0
        n_predicted = 0
        for ground_truth_boxes, predicted_boxes in zip(
            self.ground_truth_boxes_per_image, self.prediction_boxes_per_image
        ):
            n_true += len(ground_truth_boxes)
            n_predicted += len(predicted_boxes)
            if len(predicted_boxes) > 0:
                if len(ground_truth_boxes) > 0:
                    iou_matrix = get_iou_matrix(ground_truth_boxes, predicted_boxes)
                    n_false_negatives += get_n_false_negatives(iou_matrix, iou_threshold)
            else:
                n_false_negatives += len(ground_truth_boxes)
        return _ResultCounters(n_false_negatives, n_true, n_predicted)


class FMeasure(IPerformanceProvider):
    """Computes the f-measure (also known as F1-score) for a resultset.

    The f-measure is typically used in detection (localization) tasks to obtain a single number that balances precision
    and recall.

    To determine whether a predicted box matches a ground truth box an overlap measured
    is used based on a minimum
    intersection-over-union (IoU), by default a value of 0.5 is used.

    In addition spurious results are eliminated by applying non-max suppression (NMS) so that two predicted boxes with
    IoU > threshold are reduced to one. This threshold can be determined automatically by setting `vary_nms_threshold`
    to True.

    Args:
        resultset (ResultSetEntity) :ResultSet entity used for calculating the F-Measure
        vary_confidence_threshold (bool): if True the maximal F-measure is determined by optimizing for different
            confidence threshold values Defaults to False.
        vary_nms_threshold (bool): if True the maximal F-measure is determined by optimizing for different NMS threshold
            values. Defaults to False.
        cross_class_nms (bool): Whether non-max suppression should be applied cross-class. If True this will eliminate
            boxes with sufficient overlap even if they are from different classes. Defaults to False.

    Raises:
        ValueError: if prediction dataset and ground truth dataset are empty
    """

    def __init__(
        self,
        resultset: ResultSetEntity,
        vary_confidence_threshold: bool = False,
        vary_nms_threshold: bool = False,
        cross_class_nms: bool = False,
    ):

        ground_truth_dataset: DatasetEntity = resultset.ground_truth_dataset
        prediction_dataset: DatasetEntity = resultset.prediction_dataset

        if len(prediction_dataset) == 0 or len(ground_truth_dataset) == 0:
            raise ValueError("Cannot compute the F-measure of an empty result set.")

        labels = resultset.model.configuration.get_label_schema().get_labels(include_empty=False)
        classes = [label.name for label in labels]
        boxes_pair = _FMeasureCalculator(
            FMeasure.__get_boxes_from_dataset_as_list(ground_truth_dataset, labels),
            FMeasure.__get_boxes_from_dataset_as_list(prediction_dataset, labels),
        )
        result = boxes_pair.evaluate_detections(
            result_based_nms_threshold=vary_nms_threshold,
            classes=classes,
            cross_class_nms=cross_class_nms,
        )
        self._f_measure = ScoreMetric(name="f-measure", value=result.best_f_measure)
        self._f_measure_per_label: Dict[LabelEntity, ScoreMetric] = {}
        for label in labels:
            self.f_measure_per_label[label] = ScoreMetric(
                name=label.name, value=result.best_f_measure_per_class[label.name]
            )
        self._precision = ScoreMetric(name="Precision", value=result.per_confidence.best_f_measure_metrics.precision)
        self._recall = ScoreMetric(name="Recall", value=result.per_confidence.best_f_measure_metrics.recall)

        self._f_measure_per_confidence: Optional[CurveMetric] = None
        self._best_confidence_threshold: Optional[ScoreMetric] = None

        if vary_confidence_threshold:
            self._f_measure_per_confidence = CurveMetric(
                name="f-measure per confidence",
                xs=list(np.arange(*boxes_pair.confidence_range)),
                ys=result.per_confidence.all_classes_f_measure_curve,
            )
            self._best_confidence_threshold = ScoreMetric(
                name="Optimal confidence threshold",
                value=result.per_confidence.best_threshold,
            )

        self._f_measure_per_nms: Optional[CurveMetric] = None
        self._best_nms_threshold: Optional[ScoreMetric] = None

        if vary_nms_threshold and result.per_nms is not None:
            self._f_measure_per_nms = CurveMetric(
                name="f-measure per nms",
                xs=list(np.arange(*boxes_pair.nms_range)),
                ys=result.per_nms.all_classes_f_measure_curve,
            )
            self._best_nms_threshold = ScoreMetric(name="Optimal nms threshold", value=result.per_nms.best_threshold)

    box_class_index = 4
    box_score_index = 5

    @property
    def f_measure(self) -> ScoreMetric:
        """Returns the f-measure as ScoreMetric."""
        return self._f_measure

    @property
    def f_measure_per_label(self) -> Dict[LabelEntity, ScoreMetric]:
        """Returns the f-measure per label as dictionary (Label -> ScoreMetric)."""
        return self._f_measure_per_label

    @property
    def f_measure_per_confidence(self) -> Optional[CurveMetric]:
        """Returns the curve for f-measure per confidence as CurveMetric if exists."""
        return self._f_measure_per_confidence

    @property
    def best_confidence_threshold(self) -> Optional[ScoreMetric]:
        """Returns best confidence threshold as ScoreMetric if exists."""
        return self._best_confidence_threshold

    @property
    def f_measure_per_nms(self) -> Optional[CurveMetric]:
        """Returns the curve for f-measure per nms threshold as CurveMetric if exists."""
        return self._f_measure_per_nms

    @property
    def best_nms_threshold(self) -> Optional[ScoreMetric]:
        """Returns the best NMS threshold as ScoreMetric if exists."""
        return self._best_nms_threshold

    def get_performance(self) -> MultiScorePerformance:
        """Returns the performance which consists of the F-Measure score and the dashboard metrics.

        Returns:
            MultiScorePerformance: MultiScorePerformance object containing the F-Measure scores and the dashboard metrics.
        """
        dashboard_metrics: List[MetricsGroup] = []
        dashboard_metrics.append(
            BarMetricsGroup(
                metrics=list(self.f_measure_per_label.values()),
                visualization_info=BarChartInfo(
                    name="F-measure per label",
                    palette=ColorPalette.LABEL,
                    visualization_type=VisualizationType.RADIAL_BAR,
                ),
            )
        )
        if self.f_measure_per_confidence is not None:
            dashboard_metrics.append(
                LineMetricsGroup(
                    metrics=[self.f_measure_per_confidence],
                    visualization_info=LineChartInfo(
                        name="F-measure per confidence",
                        x_axis_label="Confidence threshold",
                        y_axis_label="F-measure",
                    ),
                )
            )

        if self.best_confidence_threshold is not None:
            dashboard_metrics.append(
                TextMetricsGroup(
                    metrics=[self.best_confidence_threshold],
                    visualization_info=TextChartInfo(
                        name="Optimal confidence threshold",
                    ),
                )
            )

        if self.f_measure_per_nms is not None:
            dashboard_metrics.append(
                LineMetricsGroup(
                    metrics=[self.f_measure_per_nms],
                    visualization_info=LineChartInfo(
                        name="F-measure per nms",
                        x_axis_label="NMS threshold",
                        y_axis_label="F-measure",
                    ),
                )
            )

        if self.best_nms_threshold is not None:
            dashboard_metrics.append(
                TextMetricsGroup(
                    metrics=[self.best_nms_threshold],
                    visualization_info=TextChartInfo(
                        name="Optimal nms threshold",
                    ),
                )
            )
        return MultiScorePerformance(
            primary_score=self.f_measure,
            additional_scores=[self._precision, self._recall],
            dashboard_metrics=dashboard_metrics,
        )

    @staticmethod
    def __get_boxes_from_dataset_as_list(
        dataset: DatasetEntity, labels: List[LabelEntity]
    ) -> List[List[Tuple[float, float, float, float, str, float]]]:
        """Return list of boxes from dataset.

        Explanation of output shape:
            a box: [x1: float, y1, x2, y2, class: str, score: float]
            boxes_per_image: [box1, box2, …]
            ground_truth_boxes_per_image: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]

        Args:
            dataset (DatasetEntity): Dataset to get boxes from.
            labels (List[LabelEntity]): Labels to get boxes for.

        Returns:
            List[List[Tuple[float, float, float, float, str, float]]]: List of boxes for each image in the dataset.
        """
        boxes_per_image = []
        converted_types_to_box = set()
        label_names = {label.name for label in labels}
        for item in dataset:
            boxes: List[Tuple[float, float, float, float, str, float]] = []
            roi_as_box = Annotation(ShapeFactory.shape_as_rectangle(item.roi.shape), labels=[])
            for annotation in item.annotation_scene.annotations:
                shape_as_box = ShapeFactory.shape_as_rectangle(annotation.shape)
                box = shape_as_box.normalize_wrt_roi_shape(roi_as_box.shape)
                n_boxes_before = len(boxes)
                boxes.extend(
                    [
                        (box.x1, box.y1, box.x2, box.y2, label.name, label.probability)
                        for label in annotation.get_labels()
                        if label.name in label_names
                    ]
                )
                if not isinstance(annotation.shape, Rectangle) and len(boxes) > n_boxes_before:
                    converted_types_to_box.add(annotation.shape.__class__.__name__)
            boxes_per_image.append(boxes)
        if len(converted_types_to_box) > 0:
            logger.warning(
                f"The shapes of types {tuple(converted_types_to_box)} have been converted to their "
                f"full enclosing Box representation in order to compute the f-measure"
            )

        return boxes_per_image
