# Copyright (C) 2020-2021 Intel Corporation
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

import datetime
from typing import cast

import numpy as np
import pytest

from otx.api.configuration import ConfigurableParameters
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelSchemaEntity
from otx.api.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    ColorPalette,
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    Performance,
    ScoreMetric,
    TextChartInfo,
    TextMetricsGroup,
    VisualizationType,
)
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.evaluation.f_measure import (
    FMeasure,
    _AggregatedResults,
    _FMeasureCalculator,
    _Metrics,
    _OverallResults,
    _ResultCounters,
    bounding_box_intersection_over_union,
    get_iou_matrix,
    get_n_false_negatives,
    intersection_box,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestFMeasureFunctions:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_intersection_box(self):
        """
        <b>Description:</b>
        Check "intersection_box" function

        <b>Input data:</b>
        Bounding boxes coordinates

        <b>Expected results:</b>
        Test passes if box returned by "intersection_box" function has expected coordinates

        <b>Steps</b>
        1. Check box returned by "intersection_box" function for boxes that intersect in two points
        2. Check box returned by "intersection_box" function for boxes that intersect in one point
        3. Check box returned by "intersection_box" function for boxes that intersect by one side
        4. Check box returned by "intersection_box" function when one of boxes completely covers other
        """
        base_box = [2, 2, 5, 6]
        # Checking box returned by "intersection_box" for boxes that intersect in two points
        assert intersection_box(box1=base_box, box2=[4, 4, 7, 8]) == (4, 5, 6, 4)
        # Checking box returned by "intersection_box" for boxes that intersect in one point
        assert intersection_box(box1=base_box, box2=[1, 1, 2, 2]) == (2, 2, 2, 2)
        # Checking box returned by "intersection_box" for boxes that intersect by one side
        assert intersection_box(box1=base_box, box2=[2, 1, 5, 2]) == (2, 5, 2, 2)
        # Checking box returned by "intersection_box" when one of boxes completely covers other
        assert intersection_box(box1=base_box, box2=[0, 0, 10, 10]) == (2, 5, 6, 2)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_bounding_box_intersection_over_union(self):
        """
        <b>Description:</b>
        Check "bounding_box_intersection_over_union" function

        <b>Input data:</b>
        Bounding boxes coordinates

        <b>Expected results:</b>
        Test passes if value returned by "bounding_box_intersection_over_union" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "bounding_box_intersection_over_union" function when "x_right" coordinate of
        intersection box more than "x_left"
        2. Check value returned by "bounding_box_intersection_over_union" function when "y_bottom" coordinate of
        intersection box more than "y_top"
        3. Check value returned by "bounding_box_intersection_over_union" function when boxes intersect in two points
        """
        # Checking value returned by "bounding_box_intersection_over_union" when "x_right" coordinate of
        # intersection box more than "x_left"
        assert bounding_box_intersection_over_union(box1=[2, 2, 5, 6], box2=[7, 4, 4, 8]) == 0.0
        # Checking value returned by "bounding_box_intersection_over_union" when "y_bottom" coordinate of
        # intersection box more than "y_top"
        assert bounding_box_intersection_over_union(box1=[2, 8, 6, 1], box2=[1, 7, 5, 2]) == 0.0
        # Checking value returned by "bounding_box_intersection_over_union" when boxes intersect in two points
        assert bounding_box_intersection_over_union(box1=[1, 3, 3, 7], box2=[2, 4, 5, 5]) == 0.1

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_get_iou_matrix(self):
        """
        <b>Description:</b>
        Check "get_iou_matrix" function

        <b>Input data:</b>
        Bounding boxes coordinates

        <b>Expected results:</b>
        Test passes if array returned by "get_iou_matrix" function is equal to expected
        """
        boxes_1 = [[2, 2, 5, 6], [2, 8, 6, 1], [1, 3, 3, 7]]
        boxes_2 = [[7, 4, 4, 8], [2, 4, 5, 5], [1, 1, 2, 2], [0, 0, 10, 10]]
        expected_matrix = [
            [0.0, 0.25, 0.0, 0.12],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.08],
        ]
        assert np.array_equal(get_iou_matrix(boxes_1, boxes_2), expected_matrix)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_get_n_false_negatives(self):
        """
        <b>Description:</b>
        Check "get_n_false_negatives" function

        <b>Input data:</b>
        IoU-matrix np.array

        <b>Expected results:</b>
        Test passes if value returned by "get_n_false_negatives" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "get_n_false_negatives" function when max element in a row is less than
        "iou_threshold" parameter
        2. Check value returned by "get_n_false_negatives" function when several elements in a column are more than
        "iou_threshold" parameter
        """
        iou_matrix = np.array([[0.0, 0.25, 0.0, 0.09], [0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.08]])
        # Checking value returned by "get_n_false_negatives" when max element in a row is less than "iou_threshold"
        assert get_n_false_negatives(iou_matrix, 0.11) == 2
        # Checking value returned by "get_n_false_negatives" when several elements in a column are more than
        # "iou_threshold"
        assert get_n_false_negatives(iou_matrix, 0.09) == 2


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestMetrics:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_metrics_initialization(self):
        """
        <b>Description:</b>
        Check "_Metrics" class object initialization

        <b>Input data:</b>
        "_Metrics" class object with specified "f_measure", "precision" and "recall" parameters

        <b>Expected results:</b>
        Test passes if "f_measure", "precision" and "recall" attributes of initialized "_Metrics" class object
        are equal to expected
        """
        f_measure = 0.4
        precision = 0.9
        recall = 0.3
        metrics = _Metrics(f_measure=f_measure, precision=precision, recall=recall)
        assert metrics.f_measure == f_measure
        assert metrics.precision == precision
        assert metrics.recall == recall


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestResultCounters:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_counters_initialization(self):
        """
        <b>Description:</b>
        Check "_ResultCounters" class object initialization

        <b>Input data:</b>
        "_ResultCounters" class object with specified "n_false_negatives", "n_true" and "n_predicted" parameters

        <b>Expected results:</b>
        Test passes if "n_false_negatives", "n_true" and "n_predicted" attributes of initialized "_ResultCounters"
        class object are equal to expected
        """
        n_false_negatives = 2
        n_true = 9
        n_predicted = 9
        result_counters = _ResultCounters(n_false_negatives=n_false_negatives, n_true=n_true, n_predicted=n_predicted)
        assert result_counters.n_false_negatives == n_false_negatives
        assert result_counters.n_true == n_true
        assert result_counters.n_predicted == n_predicted

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_counters_calculate_f_measure(self):
        """
        <b>Description:</b>
        Check "_ResultCounters" class "calculate_f_measure" method

        <b>Input data:</b>
        "_ResultCounters" class object with specified "n_false_negatives", "n_true" and "n_predicted" parameters

        <b>Expected results:</b>
        Test passes if value returned by "calculate_f_measure" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "calculate_f_measure" method when "n_true" attribute of "_ResultCounters" object is
        more than "n_false_negatives"
        2. Check value returned by "calculate_f_measure" method when "n_true" attribute of "_ResultCounters" object is
        less than "n_false_negatives"
        3. Check value returned by "calculate_f_measure" method when "n_true" attribute of "_ResultCounters" object is
        equal to "n_false_negatives"
        4. Check value returned by "calculate_f_measure" method when "n_predicted" attribute of "_ResultCounters" object
        is equal to 0
        5. Check value returned by "calculate_f_measure" method when "n_true" attribute of "_ResultCounters" object is
        equal to 0
        """

        def check_calculated_f_measure(
            f_measure_actual: _Metrics,
            expected_precision: float,
            expected_recall: float,
        ):
            expected_f_measure = (2 * expected_precision * expected_recall) / (
                expected_precision + expected_recall + np.finfo(float).eps
            )
            assert f_measure_actual.f_measure == expected_f_measure
            assert f_measure_actual.precision == expected_precision
            assert f_measure_actual.recall == expected_recall

        # Checking value returned by "calculate_f_measure" when "n_true" is more than "n_false_negatives"
        result_counters = _ResultCounters(n_false_negatives=4, n_true=8, n_predicted=16)
        precision = (8 - 4) / 16  # (n_true-n_false_negatives)/n_predicted
        recall = (8 - 4) / 8  # (n_true-n_false_negatives)/n_true
        f_measure = result_counters.calculate_f_measure()
        check_calculated_f_measure(f_measure, precision, recall)
        # Checking value returned by "calculate_f_measure" when "n_true" is less than "n_false_negatives"
        result_counters = _ResultCounters(n_false_negatives=16, n_true=8, n_predicted=16)
        f_measure = result_counters.calculate_f_measure()
        precision = (8 - 16) / 16  # (n_true-n_false_negatives)/n_predicted
        recall = (8 - 16) / 8  # (n_true-n_false_negatives)/n_true
        check_calculated_f_measure(f_measure, precision, recall)
        # Checking value returned by "calculate_f_measure" when "n_true" is equal to "n_false_negatives"
        result_counters = _ResultCounters(n_false_negatives=8, n_true=8, n_predicted=14)
        f_measure = result_counters.calculate_f_measure()
        check_calculated_f_measure(f_measure, 0.0, 0.0)
        # Checking value returned by "calculate_f_measure" when "n_predicted" is equal to 0
        result_counters = _ResultCounters(n_false_negatives=2, n_true=8, n_predicted=0)
        f_measure = result_counters.calculate_f_measure()
        check_calculated_f_measure(f_measure, 1.0, 0.0)
        # Checking value returned by "calculate_f_measure" method when "n_true" is equal to 0
        result_counters = _ResultCounters(n_false_negatives=2, n_true=0, n_predicted=8)
        f_measure = result_counters.calculate_f_measure()
        check_calculated_f_measure(f_measure, 0.0, 1.0)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestAggregatedResults:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_aggregate_results_initialization(self):
        """
        <b>Description:</b>
        Check "_AggregatedResults" class object initialization

        <b>Input data:</b>
        "_AggregatedResults" class object with specified "classes" parameter

        <b>Expected results:</b>
        Test passes if attributes of initialized "_AggregatedResults" class object are equal to expected

        <b>Steps</b>
        1. Check attributes of "_AggregatedResults" object initialized with "classes" attribute equal to list with
        several classes
        2. Check attributes of "_AggregatedResults" object initialized with "classes" attribute equal to list with one
        class
        3. Check attributes of "_AggregatedResults" object initialized with "classes" attribute equal to empty list
        """

        def check_aggregate_results_attributes(aggregate_results, expected_classes_curve):
            assert aggregate_results.f_measure_curve == expected_classes_curve
            assert aggregate_results.precision_curve == expected_classes_curve
            assert aggregate_results.recall_curve == expected_classes_curve
            assert aggregate_results.all_classes_f_measure_curve == []
            assert aggregate_results.best_f_measure == 0.0
            assert aggregate_results.best_threshold == 0.0

        # Checking attributes of "_AggregatedResults" object initialized with "classes" equal to list with
        # several classes
        check_aggregate_results_attributes(
            aggregate_results=_AggregatedResults(["class_1", "class_2", "class_3"]),
            expected_classes_curve={"class_1": [], "class_2": [], "class_3": []},
        )
        # Checking attributes of "_AggregatedResults" object initialized with "classes" equal to list with one class
        check_aggregate_results_attributes(
            aggregate_results=_AggregatedResults(["class_1"]),
            expected_classes_curve={"class_1": []},
        )
        # Checking attributes of "_AggregatedResults" object initialized with "classes" equal to empty list
        check_aggregate_results_attributes(aggregate_results=_AggregatedResults([]), expected_classes_curve={})


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestOverallResults:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_counters_initialization(self):
        """
        <b>Description:</b>
        Check "_OverallResults" class object initialization

        <b>Input data:</b>
        "_OverallResults" class object with specified "per_confidence", "per_nms", "best_f_measure_per_class" and
        "best_f_measure" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized "_OverallResults" class object are equal to expected
        """
        per_confidence = _AggregatedResults(["confidence_class_1", "confidence_class_2"])
        per_nms = _AggregatedResults(["nms_class_1", "nms_class_2"])
        best_f_measure_per_class = {"class_1": 0.6, "class_2": 0.8}
        best_f_measure = 0.8
        overall_results = _OverallResults(
            per_confidence=per_confidence,
            per_nms=per_nms,
            best_f_measure_per_class=best_f_measure_per_class,
            best_f_measure=best_f_measure,
        )
        assert overall_results.per_confidence == per_confidence
        assert overall_results.per_nms == per_nms
        assert overall_results.best_f_measure_per_class == best_f_measure_per_class
        assert overall_results.best_f_measure == best_f_measure


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestFMeasureCalculator:
    @staticmethod
    def ground_truth_boxes_per_image():
        return [  # images
            [  # image_1 boxes
                (0.5, 0.1, 0.8, 0.9, "class_1", 0.95),
                (0.1, 0.1, 0.3, 0.7, "class_2", 0.93),
                (0.05, 0.05, 0.3, 0.75, "class_2", 0.91),
            ],
            [  # image_2 boxes
                (0.15, 0.0, 0.4, 0.75, "class_1", 0.9),
                (0.1, 0.0, 0.45, 0.7, "class_1", 0.88),
                (0.45, 0.0, 0.95, 0.45, "class_3", 0.94),
                (0.45, 0.0, 1.0, 0.5, "class_3", 0.92),
            ],
        ]

    @staticmethod
    def prediction_boxes_per_image():
        return [  # images
            [  # image_1 boxes
                (0.45, 0.2, 0.75, 0.85, "class_1", 0.92),
                (0.5, 0.05, 0.8, 0.85, "class_1", 0.93),
                (0.1, 0.15, 0.35, 0.75, "class_2", 0.92),
                (0.05, 0.05, 0.35, 0.7, "class_2", 0.91),
            ],
            [  # image_2 boxes
                (0.15, 0.05, 0.45, 0.8, "class_1", 0.89),
                (0.15, 0.0, 0.5, 0.7, "class_1", 0.85),
                (0.5, 0.0, 0.9, 0.5, "class_3", 0.95),
                (0.45, 0.05, 0.95, 0.5, "class_3", 0.94),
            ],
        ]

    def f_measure_calculator(self):
        return _FMeasureCalculator(
            ground_truth_boxes_per_image=self.ground_truth_boxes_per_image(),
            prediction_boxes_per_image=self.prediction_boxes_per_image(),
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_initialization(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class object initialization

        <b>Input data:</b>
        "_FMeasureCalculator" class object with specified "ground_truth_boxes_per_image" and
        "prediction_boxes_per_image" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized "_FMeasureCalculator" class object are equal to expected
        """
        ground_truth_boxes_per_image = self.ground_truth_boxes_per_image()
        prediction_boxes_per_image = self.prediction_boxes_per_image()
        f_measure_calculator = _FMeasureCalculator(
            ground_truth_boxes_per_image=ground_truth_boxes_per_image,
            prediction_boxes_per_image=prediction_boxes_per_image,
        )
        assert f_measure_calculator.ground_truth_boxes_per_image == ground_truth_boxes_per_image
        assert f_measure_calculator.prediction_boxes_per_image == prediction_boxes_per_image
        assert f_measure_calculator.confidence_range == [0.025, 1.0, 0.025]
        assert f_measure_calculator.nms_range == [0.1, 1, 0.05]
        assert f_measure_calculator.default_confidence_threshold == 0.35

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_get_counters(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "get_counters" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object with specified "ground_truth_boxes_per_image" and
        "prediction_boxes_per_image" parameters

        <b>Expected results:</b>
        Test passes if "_ResultCounters" object returned by "get_counters" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "get_counters" method for "ground_truth_boxes" and "predicted_boxes" attributes with
        length is more than 0
        2. Check value returned by "get_counters" method for "predicted_boxes" attribute with length is more than 0 and
        "ground_truth_boxes" attribute with length is equal to 0
        3. Check value returned by "get_counters" method for "ground_truth_boxes" attribute with length is more than 0
        and "predicted_boxes" attribute with length is equal to 0
        """

        def check_get_counters(
            calculator: _FMeasureCalculator,
            expected_n_false_negatives: int,
            expected_n_predicted: int,
            expected_n_true: int,
        ):
            get_counters = calculator.get_counters(0.75)
            assert isinstance(get_counters, _ResultCounters)
            assert get_counters.n_false_negatives == expected_n_false_negatives
            assert get_counters.n_predicted == expected_n_predicted
            assert get_counters.n_true == expected_n_true

        ground_boxes = self.ground_truth_boxes_per_image()
        predicted_boxes = self.prediction_boxes_per_image()
        # Checking value returned by "get_counters" for "ground_truth_boxes" and "predicted_boxes" with length is more
        # than 0
        f_measure_calculator = _FMeasureCalculator(
            ground_truth_boxes_per_image=ground_boxes,
            prediction_boxes_per_image=predicted_boxes,
        )
        check_get_counters(
            f_measure_calculator,
            expected_n_false_negatives=3,
            expected_n_predicted=8,
            expected_n_true=7,
        )
        # Checking value returned by "get_counters" for "predicted_boxes" with length is more than 0 and
        # "ground_truth_boxes" with length is equal to 0
        f_measure_calculator = _FMeasureCalculator(
            ground_truth_boxes_per_image=[[]],
            prediction_boxes_per_image=predicted_boxes,
        )
        check_get_counters(
            f_measure_calculator,
            expected_n_false_negatives=0,
            expected_n_predicted=4,
            expected_n_true=0,
        )
        # Checking value returned by "get_counters" for "ground_truth_boxes" with length is more than 0 and
        # "predicted_boxes" with length is equal to 0
        f_measure_calculator = _FMeasureCalculator(
            ground_truth_boxes_per_image=ground_boxes, prediction_boxes_per_image=[[]]
        )
        check_get_counters(
            f_measure_calculator,
            expected_n_false_negatives=3,
            expected_n_predicted=0,
            expected_n_true=3,
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_filter_confidence(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "__filter_confidence" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object, "boxes_per_image" list and "confidence_threshold" parameter

        <b>Expected results:</b>
        Test passes if list returned by "__filter_confidence" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "__filter_confidence" method for "confidence_threshold" equal to 0.0
        2. Check value returned by "__filter_confidence" method for "confidence_threshold" equal to filter some boxes
        3. Check value returned by "__filter_confidence" method for "confidence_threshold" equal to 1.0
        """
        f_measure_calculator = self.f_measure_calculator()
        boxes_per_image = f_measure_calculator.prediction_boxes_per_image
        # Checking value returned by "__filter_confidence" for "confidence_threshold" equal to 0.0
        assert (
            f_measure_calculator._FMeasureCalculator__filter_confidence(  # type: ignore[attr-defined]
                boxes_per_image, 0.0
            )
            == boxes_per_image
        )
        # Checking value returned by "__filter_confidence" for "confidence_threshold" equal to filter some boxes
        assert f_measure_calculator._FMeasureCalculator__filter_confidence(  # type: ignore[attr-defined]
            boxes_per_image, 0.92
        ) == [
            [(0.5, 0.05, 0.8, 0.85, "class_1", 0.93)],
            [
                (0.5, 0.0, 0.9, 0.5, "class_3", 0.95),
                (0.45, 0.05, 0.95, 0.5, "class_3", 0.94),
            ],
        ]
        assert f_measure_calculator._FMeasureCalculator__filter_confidence(  # type: ignore[attr-defined]
            boxes_per_image, 0.93
        ) == [
            [],
            [
                (0.5, 0.0, 0.9, 0.5, "class_3", 0.95),
                (0.45, 0.05, 0.95, 0.5, "class_3", 0.94),
            ],
        ]
        # Checking value returned by "__filter_confidence" for "confidence_threshold" equal to 1.0
        assert f_measure_calculator._FMeasureCalculator__filter_confidence(  # type: ignore[attr-defined]
            boxes_per_image, 1.0
        ) == [
            [],
            [],
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_filter_class(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "__filter_class" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object, "boxes_per_image" list and classes to filter

        <b>Expected results:</b>
        Test passes if list returned by "__filter_class" method is equal to expected

        <b>Steps</b>
        1. Check list returned by "__filter_class" method to get class represented in one image
        2. Check list returned by "__filter_class" method to get class represented in several images
        3. Check list returned by "__filter_class" method to get class that is not represented in any of images
        """
        f_measure_calculator = self.f_measure_calculator()
        boxes_per_image = f_measure_calculator.prediction_boxes_per_image
        # Checking list returned by "__filter_class" to get class represented in one image
        assert f_measure_calculator._FMeasureCalculator__filter_class(  # type: ignore[attr-defined]
            boxes_per_image, "class_2"
        ) == [
            [
                (0.1, 0.15, 0.35, 0.75, "class_2", 0.92),
                (0.05, 0.05, 0.35, 0.7, "class_2", 0.91),
            ],
            [],
        ]
        # Checking list returned by "__filter_class" to get class represented in several images
        assert f_measure_calculator._FMeasureCalculator__filter_class(  # type: ignore[attr-defined]
            boxes_per_image, "class_1"
        ) == [
            [
                (0.45, 0.2, 0.75, 0.85, "class_1", 0.92),
                (0.5, 0.05, 0.8, 0.85, "class_1", 0.93),
            ],
            [
                (0.15, 0.05, 0.45, 0.8, "class_1", 0.89),
                (0.15, 0.0, 0.5, 0.7, "class_1", 0.85),
            ],
        ]
        # Checking list returned by "__filter_class" to get class that is not represented in any of images
        assert f_measure_calculator._FMeasureCalculator__filter_class(  # type: ignore[attr-defined]
            boxes_per_image, "class_6"
        ) == [
            [],
            [],
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_filter_nms(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "__filter_nms" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object, "boxes_per_image" list, "critical_nms" list and "nms_threshold" values

        <b>Expected results:</b>
        Test passes if list returned by "__filter_nms" method is equal to expected

        <b>Steps</b>
        1. Check list returned by "__filter_nms" method for "nms_threshold" parameter that not filters any boxes
        2. Check list returned by "__filter_nms" method for "nms_threshold" that filters some boxes
        3. Check list returned by "__filter_nms" method for "nms_threshold" parameter that filters all boxes
        """
        f_measure_calculator = self.f_measure_calculator()
        boxes_per_image = f_measure_calculator.prediction_boxes_per_image
        critical_nms = [[0.5, 0.55, 0.65, 0.6], [0.6, 0.55, 0.5, 0.65]]
        # Checking list returned by "__filter_nms" for "nms_threshold" that not filters any boxes
        assert (
            f_measure_calculator._FMeasureCalculator__filter_nms(  # type: ignore[attr-defined]
                boxes_per_image, critical_nms, 1.0
            )
            == boxes_per_image
        )
        # Checking list returned by "__filter_nms" for "nms_threshold" that filters some boxes
        assert f_measure_calculator._FMeasureCalculator__filter_nms(  # type: ignore[attr-defined]
            boxes_per_image, critical_nms, 0.6
        ) == [
            [
                (0.45, 0.2, 0.75, 0.85, "class_1", 0.92),
                (0.5, 0.05, 0.8, 0.85, "class_1", 0.93),
            ],
            [
                (0.15, 0.0, 0.5, 0.7, "class_1", 0.85),
                (0.5, 0.0, 0.9, 0.5, "class_3", 0.95),
            ],
        ]
        # Checking list returned by "__filter_nms" for "nms_threshold" that filters all boxes
        assert f_measure_calculator._FMeasureCalculator__filter_nms(  # type: ignore[attr-defined]
            boxes_per_image, critical_nms, 0.1
        ) == [
            [],
            [],
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_critical_nms(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "__get_critical_nms" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object, "boxes_per_image" list and "cross_class_nms" bool value

        <b>Expected results:</b>
        Test passes if list returned by "__get_critical_nms" method is equal to expected

        <b>Steps</b>
        1. Check list returned by "__get_critical_nms" method when "cross_class_nms" parameter is "False"
        2. Check list returned by "__get_critical_nms" method when "cross_class_nms" parameter is "True"
        """
        f_measure_calculator = self.f_measure_calculator()
        boxes_per_image = [  # images
            [  # image_1 boxes
                (0.1, 0.1, 0.4, 0.4, "class_1", 0.98),
                (0.2, 0.2, 0.5, 0.5, "class_1", 0.97),
                (0.6, 0.2, 0.9, 0.5, "class_2", 0.97),
                (0.6, 0.3, 1.0, 0.6, "class_3", 0.98),
            ],
            [  # image_2 boxes
                (0.1, 0.1, 0.4, 0.5, "class_2", 0.94),
                (0.2, 0.2, 0.5, 0.6, "class_2", 0.95),
                (0.6, 0.1, 0.9, 0.6, "class_3", 0.92),
                (0.7, 0.1, 1.0, 0.6, "class_4", 0.94),
            ],
        ]
        # Checking list returned by "__get_critical_nms" when "cross_class_nms" is "False"
        assert f_measure_calculator._FMeasureCalculator__get_critical_nms(  # type: ignore[attr-defined]
            boxes_per_image, False
        ) == [
            [0.0, 0.28571428571428575, 0.0, 0.0],
            [0.3333333333333333, 0.0, 0.0, 0.0],
        ]
        # Checking list returned by "__get_critical_nms" when "cross_class_nms" is "True"
        assert f_measure_calculator._FMeasureCalculator__get_critical_nms(  # type: ignore[attr-defined]
            boxes_per_image, True
        ) == [
            [0.0, 0.28571428571428575, 0.4, 0.0],
            [0.3333333333333333, 0.0, 0.5000000000000001, 0.0],
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_get_f_measure_for_class(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "get_f_measure_for_class" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object with specified "ground_truth_boxes_per_image" and
        "prediction_boxes_per_image" parameters

        <b>Expected results:</b>
        Test passes if tuple returned by "get_f_measure_for_class" method is equal to expected

        <b>Steps</b>
        1. Check tuple returned by "get_f_measure_for_class" method for class included in _FMeasureCalculator object
        2. Check tuple returned by "get_f_measure_for_class" method for class non-included in _FMeasureCalculator object
        3. Check tuple returned by "get_f_measure_for_class" method for _FMeasureCalculator object with
        "ground_truth_boxes_per_image" parameter equal to empty list
        """

        def check_get_f_measure(actual_f_measure_for_class, expected_values_dict: dict):
            # Checking _Metrics object
            metrics = actual_f_measure_for_class[0]
            assert isinstance(metrics, _Metrics)
            assert metrics.f_measure == expected_values_dict.get("f_measure")
            assert metrics.precision == expected_values_dict.get("precision")
            assert metrics.recall == expected_values_dict.get("recall")
            # Checking _ResultCounters object
            result_counters = actual_f_measure_for_class[1]
            assert isinstance(result_counters, _ResultCounters)
            assert result_counters.n_false_negatives == expected_values_dict.get("n_false_negatives")
            assert result_counters.n_predicted == expected_values_dict.get("n_predicted")
            assert result_counters.n_true == expected_values_dict.get("n_true")

        f_measure_calculator = self.f_measure_calculator()
        # Checking tuple returned by "get_f_measure_for_class" method for class included in _FMeasureCalculator object
        check_get_f_measure(
            actual_f_measure_for_class=f_measure_calculator.get_f_measure_for_class("class_1", 0.75, 0.9),
            expected_values_dict={
                "f_measure": 0.3999999999999999,
                "precision": 0.5,
                "recall": 0.3333333333333333,
                "n_false_negatives": 2,
                "n_predicted": 2,
                "n_true": 3,
            },
        )
        # Checking tuple returned by "get_f_measure_for_class" method for non-included class in
        # _FMeasureCalculator object
        check_get_f_measure(
            actual_f_measure_for_class=f_measure_calculator.get_f_measure_for_class("class_6", 0.75, 0.9),
            expected_values_dict={
                "f_measure": 0.0,
                "precision": 1.0,
                "recall": 0.0,
                "n_false_negatives": 0,
                "n_predicted": 0,
                "n_true": 0,
            },
        )
        # Checking tuple returned by "get_f_measure_for_class" method for _FMeasureCalculator object with empty list
        # of ground_truth_boxes_per_image
        f_measure_calculator.ground_truth_boxes_per_image = []
        check_get_f_measure(
            actual_f_measure_for_class=f_measure_calculator.get_f_measure_for_class("class_1", 0.75, 0.9),
            expected_values_dict={
                "f_measure": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "n_false_negatives": 0,
                "n_predicted": 0,
                "n_true": 0,
            },
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_evaluate_classes(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "evaluate_classes" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object with specified "ground_truth_boxes_per_image" and
        "prediction_boxes_per_image" parameters

        <b>Expected results:</b>
        Test passes if dictionary returned by "evaluate_classes" method is equal to expected

        <b>Steps</b>
        1. Check dictionary returned by "evaluate_classes" method when one class is specified as "classes" parameter
        2. Check dictionary returned by "evaluate_classes" method when several classes specified as "classes" parameter
        3. Check dictionary returned by "evaluate_classes" method when "All classes" is specified in "classes" parameter
        """

        def compare_metrics(actual_metric: _Metrics, expected_metric: _Metrics):
            assert actual_metric.f_measure == expected_metric.f_measure
            assert actual_metric.recall == expected_metric.recall
            assert actual_metric.precision == expected_metric.precision

        f_measure_calculator = self.f_measure_calculator()
        # Checking dictionary returned by "evaluate_classes" when one class is specified as "classes" parameter
        evaluate_classes = f_measure_calculator.evaluate_classes(["class_1"], 0.87, 0.8)
        expected_class_1 = f_measure_calculator.get_f_measure_for_class("class_1", 0.87, 0.8)[0]
        compare_metrics(evaluate_classes.get("class_1"), expected_class_1)
        compare_metrics(evaluate_classes.get("All Classes"), expected_class_1)
        # Checking dictionary returned by "evaluate_classes" when several classes specified as "classes" parameter
        evaluate_classes = f_measure_calculator.evaluate_classes(["class_1", "class_2"], 0.87, 0.8)
        expected_class_1 = f_measure_calculator.get_f_measure_for_class("class_1", 0.87, 0.8)[0]
        expected_class_2 = f_measure_calculator.get_f_measure_for_class("class_2", 0.87, 0.8)[0]
        expected_all_classes = _ResultCounters(n_false_negatives=4, n_predicted=6, n_true=5).calculate_f_measure()
        compare_metrics(evaluate_classes.get("class_1"), expected_class_1)
        compare_metrics(evaluate_classes.get("class_2"), expected_class_2)
        compare_metrics(evaluate_classes.get("All Classes"), expected_all_classes)
        # Checking dictionary returned by "evaluate_classes" when "All classes" is specified in "classes" parameter
        evaluate_classes = f_measure_calculator.evaluate_classes(["class_1", "All Classes"], 0.87, 0.8)
        compare_metrics(evaluate_classes.get("class_1"), expected_class_1)
        compare_metrics(evaluate_classes.get("All Classes"), expected_class_1)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_get_results_per_nms(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "get_results_per_nms" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object with specified "ground_truth_boxes_per_image" and
        "prediction_boxes_per_image" parameters

        <b>Expected results:</b>
        Test passes if "_AggregatedResults" object returned by "get_results_per_nms" method is equal to expected

        <b>Steps</b>
        1. Check "_AggregatedResults" object returned by "get_results_per_nms" method when "cross_class_nms" parameter
        is "False" and All Classes "f-measure" is less than "min_f_measure" parameter
        2. Check "_AggregatedResults" object returned by "get_results_per_nms" method when "cross_class_nms" parameter
        is "True" and All Classes "f-measure" is more than "min_f_measure" parameter
        """

        def check_critical_nms_per_image(
            calculator: _FMeasureCalculator,
            actual_results: _AggregatedResults,
            iou_threshold: float,
            cross_class_nms: bool,
            expected_best_f_measure: float,
            expected_best_threshold: float,
        ):
            exp_results_per_nms = _AggregatedResults(["class_1", "class_2"])
            exp_critical_nms_per_image = calculator._FMeasureCalculator__get_critical_nms(  # type: ignore[attr-defined]
                calculator.prediction_boxes_per_image, cross_class_nms
            )
            for nms_threshold in np.arange(*calculator.nms_range):
                predict_boxes_per_image_nms = calculator._FMeasureCalculator__filter_nms(  # type: ignore[attr-defined]
                    calculator.prediction_boxes_per_image,
                    exp_critical_nms_per_image,
                    nms_threshold,
                )
                boxes_pair_for_nms = _FMeasureCalculator(
                    calculator.ground_truth_boxes_per_image,
                    predict_boxes_per_image_nms,
                )
                result_point = boxes_pair_for_nms.evaluate_classes(
                    classes=["class_1", "class_2"],
                    iou_threshold=iou_threshold,
                    confidence_threshold=calculator.default_confidence_threshold,
                )
                all_classes_f_measure = result_point["All Classes"].f_measure
                exp_results_per_nms.all_classes_f_measure_curve.append(all_classes_f_measure)

                for class_name in ["class_1", "class_2"]:
                    exp_results_per_nms.f_measure_curve[class_name].append(result_point[class_name].f_measure)
                    exp_results_per_nms.precision_curve[class_name].append(result_point[class_name].precision)
                    exp_results_per_nms.recall_curve[class_name].append(result_point[class_name].recall)
            assert actual_results.all_classes_f_measure_curve == exp_results_per_nms.all_classes_f_measure_curve
            assert actual_results.f_measure_curve == exp_results_per_nms.f_measure_curve
            assert actual_results.precision_curve == exp_results_per_nms.precision_curve
            assert actual_results.recall_curve == exp_results_per_nms.recall_curve
            assert actual_results.best_f_measure == expected_best_f_measure
            assert actual_results.best_threshold == expected_best_threshold

        f_measure_calculator = self.f_measure_calculator()
        # Checking "_AggregatedResults" object returned by "get_results_per_nms" when "cross_class_nms" is "False" and
        # All Classes "f-measure" is more than "min_f_measure"
        actual_results_per_nms = f_measure_calculator.get_results_per_nms(
            classes=["class_1", "class_2"], iou_threshold=0.6, min_f_measure=0.5
        )
        check_critical_nms_per_image(
            f_measure_calculator,
            actual_results_per_nms,
            iou_threshold=0.6,
            cross_class_nms=False,
            expected_best_f_measure=0.7499999999999998,
            expected_best_threshold=0.5500000000000002,
        )
        # Checking "_AggregatedResults" object returned by "get_results_per_nms" when "cross_class_nms" is "True" and
        # All Classes "f-measure" is less than "min_f_measure"
        actual_results_per_nms = f_measure_calculator.get_results_per_nms(
            classes=["class_1", "class_2"],
            iou_threshold=0.7,
            min_f_measure=0.8,
            cross_class_nms=True,
        )
        check_critical_nms_per_image(
            f_measure_calculator,
            actual_results_per_nms,
            iou_threshold=0.7,
            cross_class_nms=True,
            expected_best_f_measure=0.8,
            expected_best_threshold=0.5,
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_get_results_per_confidence(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "get_results_per_confidence" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object with specified "ground_truth_boxes_per_image" and
        "prediction_boxes_per_image" parameters

        <b>Expected results:</b>
        Test passes if "_AggregatedResults" object returned by "get_results_per_confidence" method is equal to expected

        <b>Steps</b>
        1. Check "_AggregatedResults" object returned by "get_results_per_confidence" method when All Classes f-measure
        is more than best f-measure in results_per_confidence
        2. Check "_AggregatedResults" object returned by "get_results_per_confidence" method when All Classes f-measure
        is less than best f-measure in results_per_confidence
        """
        f_measure_calculator = self.f_measure_calculator()
        # Check "_AggregatedResults" object returned by "get_results_per_confidence" when All Classes f-measure is more
        # than best f-measure in results_per_confidence
        expected_results_per_confidence = _AggregatedResults(["class_1", "class_2"])
        for confidence_threshold in np.arange(*[0.6, 0.9]):
            result_point = f_measure_calculator.evaluate_classes(
                classes=["class_1", "class_2"],
                iou_threshold=0.7,
                confidence_threshold=confidence_threshold,
            )
            all_classes_f_measure = result_point["All Classes"].f_measure
            expected_results_per_confidence.all_classes_f_measure_curve.append(all_classes_f_measure)

            for class_name in ["class_1", "class_2"]:
                expected_results_per_confidence.f_measure_curve[class_name].append(result_point[class_name].f_measure)
                expected_results_per_confidence.precision_curve[class_name].append(result_point[class_name].precision)
                expected_results_per_confidence.recall_curve[class_name].append(result_point[class_name].recall)

        actual_results_per_confidence = f_measure_calculator.get_results_per_confidence(
            classes=["class_1", "class_2"],
            confidence_range=[0.6, 0.9],
            iou_threshold=0.7,
        )
        assert actual_results_per_confidence.all_classes_f_measure_curve == (
            expected_results_per_confidence.all_classes_f_measure_curve
        )
        assert actual_results_per_confidence.f_measure_curve == expected_results_per_confidence.f_measure_curve
        assert actual_results_per_confidence.recall_curve == expected_results_per_confidence.recall_curve
        assert actual_results_per_confidence.best_f_measure == 0.5454545454545453
        assert actual_results_per_confidence.best_threshold == 0.6
        # Check "_AggregatedResults" object returned by "get_results_per_confidence" when All Classes f-measure is less
        # than best f-measure in results_per_confidence
        actual_results_per_confidence = f_measure_calculator.get_results_per_confidence(
            classes=["class_1", "class_2"],
            confidence_range=[0.6, 0.9],
            iou_threshold=1.0,
        )
        assert actual_results_per_confidence.all_classes_f_measure_curve == [0.0]
        assert actual_results_per_confidence.f_measure_curve == {
            "class_1": [0.0],
            "class_2": [0.0],
        }
        assert actual_results_per_confidence.recall_curve == {
            "class_1": [0.0],
            "class_2": [0.0],
        }
        assert actual_results_per_confidence.best_f_measure == 0.0
        assert actual_results_per_confidence.best_threshold == 0.1

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_calculator_evaluate_detections(self):
        """
        <b>Description:</b>
        Check "_FMeasureCalculator" class "evaluate_detections" method

        <b>Input data:</b>
        "_FMeasureCalculator" class object, "boxes_per_image" list and "cross_class_nms" bool value

        <b>Expected results:</b>
        Test passes if "_OverallResults" object returned by "evaluate_detections" method is equal to expected

        <b>Steps</b>
        1. Check "_OverallResults" object returned by "evaluate_detections" method with default optional parameters
        2. Check "_OverallResults" object returned by "evaluate_detections" method with specified optional parameters
        """

        def check_evaluate_detections(
            calculator: _FMeasureCalculator,
            evaluate_detection: _OverallResults,
            iou_threshold: float = 0.5,
            result_based_nms_threshold: bool = False,
            cross_class_nms: bool = True,
        ):
            best_f_measure_per_class = {}
            results_per_confidence = calculator.get_results_per_confidence(
                classes=["class_1", "class_3"],
                confidence_range=calculator.confidence_range,
                iou_threshold=iou_threshold,
            )
            best_f_measure = results_per_confidence.best_f_measure
            for class_name in ["class_1", "class_3"]:
                best_f_measure_per_class[class_name] = max(results_per_confidence.f_measure_curve[class_name])
            results_per_nms = None
            if result_based_nms_threshold:
                results_per_nms = calculator.get_results_per_nms(
                    classes=["class_1", "class_3"],
                    iou_threshold=iou_threshold,
                    min_f_measure=results_per_confidence.best_f_measure,
                    cross_class_nms=cross_class_nms,
                )

                for class_name in ["class_1", "class_3"]:
                    best_f_measure_per_class[class_name] = max(results_per_nms.f_measure_curve[class_name])
            expected_evaluate_detection = _OverallResults(
                results_per_confidence,
                results_per_nms,
                best_f_measure_per_class,
                best_f_measure,
            )
            assert isinstance(evaluate_detection, _OverallResults)
            assert evaluate_detection.best_f_measure == expected_evaluate_detection.best_f_measure
            assert evaluate_detection.best_f_measure_per_class == expected_evaluate_detection.best_f_measure_per_class
            assert evaluate_detection.per_confidence.all_classes_f_measure_curve == (
                expected_evaluate_detection.per_confidence.all_classes_f_measure_curve
            )
            assert evaluate_detection.per_confidence.best_f_measure == (
                expected_evaluate_detection.per_confidence.best_f_measure
            )
            assert evaluate_detection.per_confidence.best_threshold == (
                expected_evaluate_detection.per_confidence.best_threshold
            )
            assert evaluate_detection.per_confidence.f_measure_curve == (
                expected_evaluate_detection.per_confidence.f_measure_curve
            )
            assert evaluate_detection.per_confidence.precision_curve == (
                expected_evaluate_detection.per_confidence.precision_curve
            )
            assert evaluate_detection.per_confidence.recall_curve == (
                expected_evaluate_detection.per_confidence.recall_curve
            )

        f_measure_calculator = self.f_measure_calculator()
        # Checking "_OverallResults" object returned by "evaluate_detections" with default optional parameters
        actual_evaluate_detection = f_measure_calculator.evaluate_detections(["class_1", "class_3"])
        check_evaluate_detections(
            calculator=f_measure_calculator,
            evaluate_detection=actual_evaluate_detection,
        )
        # Checking "_OverallResults" object returned by "evaluate_detections" with specified optional parameters
        actual_evaluate_detection = f_measure_calculator.evaluate_detections(["class_1", "class_3"], 0.79, True, True)
        check_evaluate_detections(
            calculator=f_measure_calculator,
            evaluate_detection=actual_evaluate_detection,
            iou_threshold=0.79,
            result_based_nms_threshold=True,
            cross_class_nms=True,
        )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestFMeasure:
    color = Color(0, 255, 0)
    creation_date = datetime.datetime(year=2021, month=12, day=23)

    @staticmethod
    def generate_random_image() -> Image:
        image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))
        return image

    def model_labels(self):
        class_1_label = LabelEntity(
            name="class_1",
            domain=Domain.DETECTION,
            color=self.color,
            creation_date=self.creation_date,
        )
        class_2_label = LabelEntity(
            name="class_2",
            domain=Domain.DETECTION,
            color=self.color,
            creation_date=self.creation_date,
        )
        class_3_label = LabelEntity(
            name="class_3",
            domain=Domain.DETECTION,
            color=self.color,
            creation_date=self.creation_date,
        )
        return [class_1_label, class_2_label, class_3_label]

    def model(self):
        configurable_params = ConfigurableParameters(header="Test model configurable params")

        model_label_group = LabelGroup(name="model_labels", labels=self.model_labels(), id=ID("model_label"))

        model_configuration = ModelConfiguration(
            configurable_params, LabelSchemaEntity(label_groups=[model_label_group])
        )

        model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)
        return model

    def roi(self):
        return Annotation(
            shape=Rectangle(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="image_roi",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("image_roi"),
                    )
                )
            ],
        )

    def image_1_ground_boxes(self):
        box_1_annotation = Annotation(
            shape=Rectangle(x1=0.5, y1=0.1, x2=0.8, y2=0.9),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_1",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_1_image_1"),
                    ),
                    probability=0.95,
                )
            ],
        )

        box_2_annotation = Annotation(
            shape=Rectangle(x1=0.1, y1=0.1, x2=0.3, y2=0.7),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_2",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_2_image_1"),
                    ),
                    probability=0.93,
                )
            ],
        )

        box_3_annotation = Annotation(
            shape=Rectangle(x1=0.05, y1=0.05, x2=0.3, y2=0.75),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_2",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_2_image_1"),
                    ),
                    probability=0.91,
                )
            ],
        )

        annotation_scene = AnnotationSceneEntity(
            annotations=[box_1_annotation, box_2_annotation, box_3_annotation],
            kind=AnnotationSceneKind.ANNOTATION,
        )

        image_1 = DatasetItemEntity(
            media=self.generate_random_image(),
            annotation_scene=annotation_scene,
            roi=self.roi(),
        )
        return image_1

    def image_2_ground_boxes(self):
        box_1_annotation = Annotation(
            shape=Rectangle(x1=0.15, y1=0.0, x2=0.4, y2=0.75),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_1",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_1_image_2"),
                    ),
                    probability=0.9,
                )
            ],
        )

        box_2_annotation = Annotation(
            shape=Rectangle(x1=0.1, y1=0.0, x2=0.45, y2=0.7),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_1",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_1_image_2"),
                    ),
                    probability=0.88,
                )
            ],
        )

        box_3_annotation = Annotation(
            shape=Ellipse(x1=0.45, y1=0.0, x2=0.95, y2=0.45),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_3",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_3_image_2"),
                    ),
                    probability=0.94,
                )
            ],
        )

        box_4_annotation = Annotation(
            shape=Ellipse(x1=0.45, y1=0.0, x2=1.0, y2=0.5),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_3",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_3_image_2"),
                    ),
                    probability=0.92,
                )
            ],
        )

        annotation_scene = AnnotationSceneEntity(
            annotations=[
                box_1_annotation,
                box_2_annotation,
                box_3_annotation,
                box_4_annotation,
            ],
            kind=AnnotationSceneKind.ANNOTATION,
        )

        image_2 = DatasetItemEntity(
            media=self.generate_random_image(),
            annotation_scene=annotation_scene,
            roi=self.roi(),
        )
        return image_2

    def image_1_prediction_boxes(self):
        box_1_annotation = Annotation(
            shape=Rectangle(x1=0.45, y1=0.2, x2=0.75, y2=0.85),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_1",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_1_image_1"),
                    ),
                    probability=0.92,
                )
            ],
        )

        box_2_annotation = Annotation(
            shape=Rectangle(x1=0.5, y1=0.05, x2=0.8, y2=0.85),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_1",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_1_image_1"),
                    ),
                    probability=0.93,
                )
            ],
        )

        box_3_annotation = Annotation(
            shape=Rectangle(x1=0.1, y1=0.15, x2=0.35, y2=0.75),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_2",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_2_image_1"),
                    ),
                    probability=0.92,
                )
            ],
        )

        box_4_annotation = Annotation(
            shape=Rectangle(x1=0.05, y1=0.05, x2=0.35, y2=0.7),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_2",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_2_image_1"),
                    ),
                    probability=0.91,
                )
            ],
        )

        annotation_scene = AnnotationSceneEntity(
            annotations=[
                box_1_annotation,
                box_2_annotation,
                box_3_annotation,
                box_4_annotation,
            ],
            kind=AnnotationSceneKind.ANNOTATION,
        )

        image_1 = DatasetItemEntity(
            media=self.generate_random_image(),
            annotation_scene=annotation_scene,
            roi=self.roi(),
        )
        return image_1

    def image_2_prediction_boxes(self):
        box_1_annotation = Annotation(
            shape=Rectangle(x1=0.15, y1=0.05, x2=0.45, y2=0.8),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_1",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_1_image_2"),
                    ),
                    probability=0.89,
                )
            ],
        )

        box_2_annotation = Annotation(
            shape=Rectangle(x1=0.15, y1=0.0, x2=0.5, y2=0.7),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_1",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_1_image_2"),
                    ),
                    probability=0.85,
                )
            ],
        )

        box_3_annotation = Annotation(
            shape=Ellipse(x1=0.5, y1=0.0, x2=0.9, y2=0.5),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_3",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_3_image_2"),
                    ),
                    probability=0.95,
                )
            ],
        )

        box_4_annotation = Annotation(
            shape=Ellipse(x1=0.45, y1=0.05, x2=0.95, y2=0.5),
            labels=[
                ScoredLabel(
                    LabelEntity(
                        name="class_3",
                        domain=Domain.DETECTION,
                        color=self.color,
                        creation_date=self.creation_date,
                        id=ID("class_3_image_2"),
                    ),
                    probability=0.94,
                )
            ],
        )

        annotation_scene = AnnotationSceneEntity(
            annotations=[
                box_1_annotation,
                box_2_annotation,
                box_3_annotation,
                box_4_annotation,
            ],
            kind=AnnotationSceneKind.ANNOTATION,
        )

        image_2 = DatasetItemEntity(
            media=self.generate_random_image(),
            annotation_scene=annotation_scene,
            roi=self.roi(),
        )
        return image_2

    def ground_truth_dataset(self):
        return DatasetEntity([self.image_1_ground_boxes(), self.image_2_ground_boxes()])

    def prediction_dataset(self):
        return DatasetEntity([self.image_1_prediction_boxes(), self.image_2_prediction_boxes()])

    def incorrect_prediction_dataset(self):
        return DatasetEntity([self.image_2_prediction_boxes(), self.image_2_prediction_boxes()])

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_initialization(self):
        """
        <b>Description:</b>
        Check "FMeasure" class object initialization

        <b>Input data:</b>
        "FMeasure" class object with specified "ground_truth_boxes_per_image" and "prediction_boxes_per_image"
        parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized "FMeasure" object are equal to expected

        <b>Steps</b>
        1. Check attributes of "FMeasure" class object initialized with default optional parameters
        2. Check attributes of "FMeasure" class object initialized with specified optional parameters
        3. Check ValueError exception is raised when empty list "ground_truth_dataset" is specified in "result_set"
        parameter
        4. Check ValueError exception is raised when empty list "prediction_dataset" is specified in "result_set"
        parameter
        """

        def check_f_measure_common_attributes(f_measure_actual):
            assert f_measure_actual.box_class_index == 4
            assert f_measure_actual.box_score_index == 5
            assert isinstance(f_measure_actual.f_measure, ScoreMetric)
            assert f_measure_actual.f_measure.name == "f-measure"
            assert f_measure_actual.f_measure.value == pytest.approx(0.2857142857142856)

        ground_dataset = self.ground_truth_dataset()
        prediction_dataset = self.prediction_dataset()
        result_set = ResultSetEntity(
            model=self.model(),
            ground_truth_dataset=ground_dataset,
            prediction_dataset=prediction_dataset,
        )
        labels = self.model_labels()
        # Checking attributes of "FMeasure" class object initialized with default optional parameters
        f_measure = FMeasure(result_set)
        check_f_measure_common_attributes(f_measure_actual=f_measure)
        assert f_measure.f_measure_per_label == {
            labels[0]: ScoreMetric(name="class_1", value=0.6666666666666665),
            labels[2]: ScoreMetric(name="class_3", value=0.0),
            labels[1]: ScoreMetric(name="class_2", value=0.0),
        }
        assert not f_measure.best_confidence_threshold
        assert not f_measure.best_nms_threshold
        assert not f_measure.f_measure_per_confidence
        assert not f_measure.f_measure_per_nms
        # Checking attributes of "FMeasure" class object initialized with specified optional parameters
        f_measure = FMeasure(
            resultset=result_set,
            vary_confidence_threshold=True,
            vary_nms_threshold=True,
            cross_class_nms=True,
        )
        check_f_measure_common_attributes(f_measure_actual=f_measure)
        assert f_measure.f_measure_per_label == {
            labels[0]: ScoreMetric(name="class_1", value=0.7999999999999999),
            labels[1]: ScoreMetric(name="class_2", value=0.6666666666666665),
            labels[2]: ScoreMetric(name="class_3", value=0.6666666666666665),
        }
        label_schema_labels = result_set.model.configuration.get_label_schema().get_labels(include_empty=False)
        classes = [label.name for label in label_schema_labels]
        boxes_pair = _FMeasureCalculator(
            f_measure._FMeasure__get_boxes_from_dataset_as_list(ground_dataset, labels),  # type: ignore[attr-defined]
            f_measure._FMeasure__get_boxes_from_dataset_as_list(  # type: ignore[attr-defined]
                prediction_dataset, labels
            ),
        )
        result = boxes_pair.evaluate_detections(
            result_based_nms_threshold=True,
            classes=classes,
            cross_class_nms=True,
        )
        expected_f_measure_per_confidence = CurveMetric(
            name="f-measure per confidence",
            xs=list(np.arange(*boxes_pair.confidence_range)),
            ys=result.per_confidence.all_classes_f_measure_curve,
        )
        assert f_measure.f_measure_per_confidence.name == expected_f_measure_per_confidence.name
        assert f_measure.f_measure_per_confidence.xs == expected_f_measure_per_confidence.xs
        assert f_measure.f_measure_per_confidence.ys == expected_f_measure_per_confidence.ys
        expected_f_measure_per_nms = CurveMetric(
            name="f-measure per nms",
            xs=list(np.arange(*boxes_pair.nms_range)),
            ys=result.per_nms.all_classes_f_measure_curve,
        )
        assert f_measure.f_measure_per_nms.name == expected_f_measure_per_nms.name
        assert f_measure.f_measure_per_nms.xs == expected_f_measure_per_nms.xs
        assert f_measure.f_measure_per_nms.ys == expected_f_measure_per_nms.ys
        # Checking ValueError exception is raised when empty list "ground_truth_dataset" is specified in "result_set"
        empty_dataset = DatasetEntity([])
        result_set = ResultSetEntity(
            model=self.model(),
            ground_truth_dataset=empty_dataset,
            prediction_dataset=prediction_dataset,
        )
        with pytest.raises(ValueError):
            FMeasure(result_set)
        # Checking ValueError exception is raised when empty list "prediction_dataset" is specified in "result_set"
        result_set = ResultSetEntity(
            model=self.model(),
            ground_truth_dataset=ground_dataset,
            prediction_dataset=empty_dataset,
        )
        with pytest.raises(ValueError):
            FMeasure(result_set)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_f_measure_get_performance(self):
        """
        <b>Description:</b>
        Check "FMeasure" class "get_performance" method

        <b>Input data:</b>
        "FMeasure" class object with specified "ground_truth_boxes_per_image" and "prediction_boxes_per_image"
        parameters

        <b>Expected results:</b>
        Test passes if "Performance" object returned by "get_performance" method is equal to expected

        <b>Steps</b>
        1. Check "Performance" object returned by "get_performance" method for "FMeasure" object initialized with
        default optional parameters
        2. Check "Performance" object returned by "get_performance" method for "FMeasure" object initialized with
        specified optional parameters
        """

        def check_performance(performance, expected_score, expected_metric_groups):
            assert isinstance(performance, Performance)
            assert isinstance(performance.score, ScoreMetric)
            assert performance.score.name == "f-measure"
            assert performance.score.value == pytest.approx(expected_score)
            # Checking dashboard metrics
            for expected_metric_group in expected_metric_groups:
                metric_group_index = expected_metric_groups.index(expected_metric_group)
                actual_metric_group = performance.dashboard_metrics[metric_group_index]
                if isinstance(expected_metric_group, BarMetricsGroup):
                    assert actual_metric_group.metrics == expected_metric_group.metrics
                    assert isinstance(actual_metric_group.visualization_info, BarChartInfo)
                    assert actual_metric_group.visualization_info.name == "F-measure per label"
                    assert actual_metric_group.visualization_info.palette == ColorPalette.LABEL
                    assert actual_metric_group.visualization_info.type == VisualizationType.RADIAL_BAR
                if isinstance(expected_metric_group, LineMetricsGroup):
                    assert actual_metric_group.metrics == expected_metric_group.metrics
                    assert actual_metric_group.visualization_info.name == expected_metric_group.visualization_info.name
                    assert actual_metric_group.visualization_info.palette == ColorPalette.DEFAULT
                    assert (
                        actual_metric_group.visualization_info.x_axis_label
                        == expected_metric_group.visualization_info.x_axis_label
                    )
                    assert (
                        actual_metric_group.visualization_info.y_axis_label
                        == expected_metric_group.visualization_info.y_axis_label
                    )
                if isinstance(expected_metric_group, TextMetricsGroup):
                    assert actual_metric_group.metrics == expected_metric_group.metrics
                    assert actual_metric_group.visualization_info.name == expected_metric_group.visualization_info.name
                    assert actual_metric_group.visualization_info.palette == ColorPalette.DEFAULT
                    assert actual_metric_group.visualization_info.type == VisualizationType.TEXT

        def generate_expected_default_dashboard_metric_groups(
            actual_f_measure: FMeasure,
        ):
            return [
                BarMetricsGroup(
                    metrics=list(actual_f_measure.f_measure_per_label.values()),
                    visualization_info=BarChartInfo(
                        name="F-measure per label",
                        palette=ColorPalette.LABEL,
                        visualization_type=VisualizationType.RADIAL_BAR,
                    ),
                )
            ]

        def generate_expected_optional_dashboard_metric_groups(
            actual_f_measure: FMeasure,
        ):
            return [
                generate_expected_default_dashboard_metric_groups(actual_f_measure)[0],
                LineMetricsGroup(
                    metrics=[cast(CurveMetric, actual_f_measure.f_measure_per_confidence)],
                    visualization_info=LineChartInfo(
                        name="F-measure per confidence",
                        x_axis_label="Confidence threshold",
                        y_axis_label="F-measure",
                    ),
                ),
                TextMetricsGroup(
                    metrics=[cast(ScoreMetric, actual_f_measure.best_confidence_threshold)],
                    visualization_info=TextChartInfo(name="Optimal confidence threshold"),
                ),
                LineMetricsGroup(
                    metrics=[cast(CurveMetric, actual_f_measure.f_measure_per_nms)],
                    visualization_info=LineChartInfo(
                        name="F-measure per nms",
                        x_axis_label="NMS threshold",
                        y_axis_label="F-measure",
                    ),
                ),
                TextMetricsGroup(
                    metrics=[cast(ScoreMetric, actual_f_measure.best_nms_threshold)],
                    visualization_info=TextChartInfo(name="Optimal nms threshold"),
                ),
            ]

        ground_dataset = self.ground_truth_dataset()
        prediction_dataset = self.prediction_dataset()
        result_set = ResultSetEntity(
            model=self.model(),
            ground_truth_dataset=ground_dataset,
            prediction_dataset=prediction_dataset,
        )
        # Checking "Performance" object returned by "get_performance" for "FMeasure" object initialized with default
        # optional parameters
        f_measure = FMeasure(result_set)
        expected_dashboard_metric_groups = generate_expected_default_dashboard_metric_groups(f_measure)
        actual_performance = f_measure.get_performance()
        check_performance(
            performance=actual_performance,
            expected_score=0.2857142857142856,
            expected_metric_groups=expected_dashboard_metric_groups,
        )
        # Check for incorrect prediction dataset
        incorrect_prediction_dataset = self.incorrect_prediction_dataset()
        incorrect_result_set = ResultSetEntity(
            model=self.model(),
            ground_truth_dataset=ground_dataset,
            prediction_dataset=incorrect_prediction_dataset,
        )
        f_measure = FMeasure(incorrect_result_set)
        expected_dashboard_metric_groups = generate_expected_default_dashboard_metric_groups(f_measure)
        actual_performance = f_measure.get_performance()
        check_performance(
            performance=actual_performance,
            expected_score=0.15384615384615372,
            expected_metric_groups=expected_dashboard_metric_groups,
        )
        # Checking attributes of "FMeasure" class object initialized with specified values of optional parameters
        f_measure = FMeasure(
            resultset=result_set,
            vary_confidence_threshold=True,
            vary_nms_threshold=True,
            cross_class_nms=True,
        )
        expected_dashboard_metric_groups = generate_expected_optional_dashboard_metric_groups(f_measure)
        actual_performance = f_measure.get_performance()
        check_performance(
            performance=actual_performance,
            expected_score=0.2857142857142856,
            expected_metric_groups=expected_dashboard_metric_groups,
        )
        # Check for incorrect prediction dataset
        f_measure = FMeasure(
            resultset=incorrect_result_set,
            vary_confidence_threshold=True,
            vary_nms_threshold=True,
            cross_class_nms=True,
        )
        expected_dashboard_metric_groups = generate_expected_optional_dashboard_metric_groups(f_measure)
        actual_performance = f_measure.get_performance()
        check_performance(
            performance=actual_performance,
            expected_score=0.15384615384615372,
            expected_metric_groups=expected_dashboard_metric_groups,
        )
