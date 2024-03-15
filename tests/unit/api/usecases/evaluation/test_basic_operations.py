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

import numpy as np
import pytest

from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.evaluation.basic_operations import (
    divide_arrays_with_possible_zeros,
    get_intersections_and_cardinalities,
    intersection_box,
    intersection_over_union,
    precision_per_class,
    recall_per_class,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestBasicOperationsFunctions:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_get_intersections_and_cardinalities(self):
        """
        <b>Description:</b>
        Check "get_intersections_and_cardinalities" function

        <b>Input data:</b>
        "references" masks array, "predictions" masks array, "labels" list of "LabelEntity" class objects

        <b>Expected results:</b>
        Test passes if tuple returned by "get_intersections_and_cardinalities" function is equal to expected
        """
        equal_array = np.array([(1, 3, 1, 0), (2, 0, 2, 1), (3, 1, 0, 2), (0, 1, 1, 0)])
        other_equal_array = np.array([(2, 1, 1, 0), (1, 0, 2, 0), (2, 1, 0, 1)])
        unequal_reference_array = np.array([(1, 2, 3), (3, 0, 1), (1, 1, 0)])
        unequal_predictions_array = np.array([(0, 2, 3), (3, 0, 2), (1, 0, 1)])
        label_for_first_intersection = LabelEntity(name="label_for_intersection_1", domain=Domain.DETECTION)
        label_for_second_intersection = LabelEntity(name="label_for_intersection_2", domain=Domain.DETECTION)
        label_for_third_intersection = LabelEntity(name="label_for_intersection_3", domain=Domain.DETECTION)
        non_assigned_label = LabelEntity(name="non_assigned", domain=Domain.DETECTION)
        labels = [
            label_for_first_intersection,
            label_for_second_intersection,
            label_for_third_intersection,
            non_assigned_label,
        ]
        intersections_and_cardinalities = get_intersections_and_cardinalities(
            references=[equal_array, unequal_reference_array, other_equal_array],
            predictions=[equal_array, unequal_predictions_array, other_equal_array],
            labels=labels,
        )
        # Checking intersections
        intersections = intersections_and_cardinalities[0]
        assert len(intersections) == 5
        assert intersections.get(label_for_first_intersection) == 12
        assert intersections.get(label_for_second_intersection) == 7
        assert intersections.get(label_for_third_intersection) == 4
        assert intersections.get(non_assigned_label) == 0
        assert intersections.get(None) == 23
        # Checking cardinalities
        cardinalities = intersections_and_cardinalities[1]
        assert len(cardinalities) == 5
        assert cardinalities.get(label_for_first_intersection) == 28
        assert cardinalities.get(label_for_second_intersection) == 15
        assert cardinalities.get(label_for_third_intersection) == 8
        assert cardinalities.get(non_assigned_label) == 0
        assert cardinalities.get(None) == 51

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_intersection_box(self):
        """
        <b>Description:</b>
        Check "intersection_box" function

        <b>Input data:</b>
        "box_1" and "box_2" Rectangle-class objects

        <b>Expected results:</b>
        Test passes if list returned by "intersection_box" function is equal to expected

        <b>Steps</b>
        1. Check list returned by "intersection_box" function for boxes that intersect in two points
        2. Check list returned by "intersection_box" function for boxes that intersect in one point
        3. Check list returned by "intersection_box" function for boxes that intersect by one side
        4. Check list returned by "intersection_box" function when one of boxes completely overlaps other
        5. Check list returned by "intersection_box" function for boxes that not intersect
        """
        box_1 = Rectangle(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
        # Checking list returned by "intersection_box" for boxes that intersect in two points
        box_2 = Rectangle(x1=0.2, y1=0.2, x2=0.5, y2=0.5)
        assert intersection_box(box_1, box_2) == [0.2, 0.2, 0.3, 0.4]
        # Checking list returned by "intersection_box" for boxes that intersect in one point
        box_2 = Rectangle(x1=0.3, y1=0.4, x2=0.5, y2=0.5)
        assert not intersection_box(box_1, box_2)
        # Checking list returned by "intersection_box" for boxes that intersect by one side
        box_2 = Rectangle(x1=0.3, y1=0.3, x2=0.5, y2=0.4)
        assert not intersection_box(box_1, box_2)
        # Checking list returned by "intersection_box" when one of boxes completely overlaps other
        box_2 = Rectangle(x1=0.1, y1=0.1, x2=0.2, y2=0.3)
        assert intersection_box(box_1, box_2) == [0.1, 0.1, 0.2, 0.3]
        # Checking list returned by "intersection_box" for boxes that not intersect
        box_2 = Rectangle(x1=0.4, y1=0.4, x2=0.6, y2=0.7)
        assert not intersection_box(box_1, box_2)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_intersection_over_union(self):
        """
        <b>Description:</b>
        Check "intersection_over_union" function

        <b>Input data:</b>
        "box_1" and "box_2" Rectangle-class objects

        <b>Expected results:</b>
        Test passes if value returned by "intersection_over_union" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "intersection_over_union" function for boxes that intersect in two points
        2. Check value returned by "intersection_over_union" function for boxes that intersect in one point
        3. Check value returned by "intersection_over_union" function for boxes that intersect by one side
        4. Check value returned by "intersection_over_union" function when one of boxes completely overlaps other
        5. Check value returned by "intersection_over_union" function for boxes that not intersect
        """
        box_1 = Rectangle(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
        # Checking value returned by "intersection_over_union" for boxes that intersect in two points
        box_2 = Rectangle(x1=0.2, y1=0.2, x2=0.5, y2=0.5)
        assert round(intersection_over_union(box_1, box_2), 6) == 0.153846
        # Checking value returned by "intersection_over_union" for boxes that intersect in one point
        box_2 = Rectangle(x1=0.3, y1=0.4, x2=0.5, y2=0.5)
        assert intersection_over_union(box_1, box_2) == 0.0
        # Checking value returned by "intersection_over_union" for boxes that intersect by one side
        box_2 = Rectangle(x1=0.3, y1=0.3, x2=0.5, y2=0.4)
        assert intersection_over_union(box_1, box_2) == 0.0
        # Checking value returned by "intersection_over_union" when one of boxes completely overlaps other
        box_2 = Rectangle(x1=0.1, y1=0.1, x2=0.2, y2=0.3)
        assert round(intersection_over_union(box_1, box_2), 6) == 0.333333
        # Checking value returned by "intersection_over_union" for boxes that not intersect
        box_2 = Rectangle(x1=0.4, y1=0.4, x2=0.6, y2=0.7)
        assert intersection_over_union(box_1, box_2) == 0.0

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_precision_per_class(self):
        """
        <b>Description:</b>
        Check "precision_per_class" function

        <b>Input data:</b>
        Confusion matrix

        <b>Expected results:</b>
        Test passes if array returned by "precision_per_class" function is equal to expected

        <b>Steps</b>
        1. Check array returned by "precision_per_class" function for square matrix
        2. Check array returned by "precision_per_class" function for non-square matrix
        """
        # Checking array returned by "precision_per_class" for square matrix
        matrix = np.array([(0.5, 1.0, 1.0), (1.0, 0.5, 0.5), (0.5, 1.0, 1.0)])
        assert np.array_equal(precision_per_class(matrix), np.array([0.25, 0.2, 0.4]))
        # Checking array returned by "precision_per_class" for non-square matrix
        matrix = np.array([(0.6, 0.3, 0.6), (0.3, 0.6, 0.3), (0.6, 0.8, 0.3), (0.9, 0.3, 0.6)])
        assert np.array_equal(precision_per_class(matrix), np.array([0.25, 0.3]))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_recall_per_class(self):
        """
        <b>Description:</b>
        Check "recall_per_class" function

        <b>Input data:</b>
        Confusion matrix

        <b>Expected results:</b>
        Test passes if array returned by "recall_per_class" function is equal to expected
        """
        matrix = np.array([(6, 2, 0), (0, 10, 6), (0, 8, 12)])
        assert np.array_equal(recall_per_class(matrix), np.array([0.75, 0.625, 0.6]))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_divide_arrays_with_possible_zeros(self):
        """
        <b>Description:</b>
        Check "divide_arrays_with_possible_zeros" function

        <b>Input data:</b>
        "numerator" and "denominator" matrices

        <b>Expected results:</b>
        Test passes if array returned by "divide_arrays_with_possible_zeros" function is equal to expected
        """
        array = np.array([(6, 2, 0), (0, 10, 6), (0, 8, 12)])
        other_array = np.array([(2, 4, 0), (0, 2, 4), (0, 0, 1)])
        assert np.array_equal(
            a1=divide_arrays_with_possible_zeros(array, other_array),
            a2=np.array([(3, 0.5, 0), (0, 5, 1.5), (0, 0, 12)]),
        )
