# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import warnings

import numpy as np
import pytest

from ote_sdk.entities.metrics import DurationMetric, MatrixMetric
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestMetrics:
    # todo: implement tests for the metrics
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_duration_metrics(self):
        """
        <b>Description:</b>
        Check that duration is correctly calculated

        <b>Input data:</b>
        1 hour, 1 minute and 15.4 seconds

        <b>Expected results:</b>
        Test passes if DurationMetrics correctly converts seconds to hours, minutes and seconds

        <b>Steps</b>
        1. Create DurationMetrics
        2. Check hour, minute and second calculated by DurationMetric
        """
        hour = 1
        minute = 1
        second = 15.5
        seconds = (hour * 3600) + (minute * 60) + second
        duration_metric = DurationMetric.from_seconds(
            name="Training duration", seconds=seconds
        )
        assert duration_metric.hour == hour
        assert duration_metric.minute == minute
        assert duration_metric.second == second
        print(duration_metric.get_duration_string())

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_matrix_metric(self):
        """
        <b>Description:</b>
        Check that MatrixMetric correctly normalizes the values in a given matrix

        <b>Input data:</b>
        Three square matrices

        <b>Expected results:</b>
        Test passes if the values of the normalized matrices match the pre-computed matrices

        <b>Steps</b>
        1. Create Matrices
        2. Check normalized matrices against pre-computed matrices
        """
        matrix_data = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]])
        matrix_metric = MatrixMetric(
            name="test matrix", matrix_values=matrix_data, normalize=True
        )

        required_normalised_matrix_data = np.array(
            [[0, 0.5, 0.5], [0, 0.5, 0.5], [1, 0, 0]]
        )
        assert np.array_equal(
            required_normalised_matrix_data, matrix_metric.matrix_values
        )

        with warnings.catch_warnings():
            # there is a matrix with zero sum in row, so we expect 0/0 division.
            warnings.filterwarnings(
                "ignore", "invalid value encountered in true_divide"
            )
            matrix_data_with_zero_sum = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0]])
            matrix_metric_with_zero_sum = MatrixMetric(
                name="test matrix",
                matrix_values=matrix_data_with_zero_sum,
                normalize=True,
            )

        required_normalised_matrix_data_with_zero_sum = np.array(
            [[0, 0, 0], [0, 0.5, 0.5], [1, 0, 0]]
        )
        assert np.array_equal(
            required_normalised_matrix_data_with_zero_sum,
            matrix_metric_with_zero_sum.matrix_values,
        )
