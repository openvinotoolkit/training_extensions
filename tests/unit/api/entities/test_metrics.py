# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import datetime
import warnings

import numpy as np
import pytest

from otx.api.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    ColorPalette,
    CountMetric,
    CurveMetric,
    DateMetric,
    DurationMetric,
    InfoMetric,
    LineChartInfo,
    LineMetricsGroup,
    MatrixChartInfo,
    MatrixMetric,
    MatrixMetricsGroup,
    MultiScorePerformance,
    NullMetric,
    NullPerformance,
    Performance,
    ScoreMetric,
    TextChartInfo,
    TextMetricsGroup,
    VisualizationInfo,
    VisualizationType,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestMetrics:
    @staticmethod
    def mixed_conditions_duration_metric() -> DurationMetric:
        return DurationMetric(name="Mixed conditions metric", hour=0, minute=1, second=2.1)

    @staticmethod
    def matrix_data():
        return np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]])

    def normalized_matrix_metric(self) -> MatrixMetric:
        return MatrixMetric(name="test matrix", matrix_values=self.matrix_data(), normalize=True)

    @staticmethod
    def normalized_matrix_zero_sum() -> MatrixMetric:
        matrix_data_with_zero_sum = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0]])
        return MatrixMetric(name="test matrix", matrix_values=matrix_data_with_zero_sum, normalize=True)

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        3. Check value returned by get_duration_string method
        """
        hour = 1
        minute = 1
        second = 15.5
        seconds = (hour * 3600) + (minute * 60) + second
        duration_metric = DurationMetric.from_seconds(name="Training duration", seconds=seconds)
        assert duration_metric.hour == hour
        assert duration_metric.minute == minute
        assert duration_metric.second == second
        assert duration_metric.type() == "duration"
        print(duration_metric.get_duration_string())
        # Checking get_duration_string method for 0 specified as DurationMetric parameters
        zero_duration_metric = DurationMetric(name="Zero Duration Metric", hour=0, minute=0, second=0.0)
        assert zero_duration_metric.get_duration_string() == ""
        # Checking get_duration_string method for 1 specified as DurationMetric parameters
        one_duration_metric = DurationMetric(name="One Duration Metric", hour=1, minute=1, second=1.0)
        assert one_duration_metric.get_duration_string() == "1 hour 1 minute 1.00 second"
        # Checking get_duration_string method for value>1 specified as DurationMetric parameters
        more_than_one_duration_metric = DurationMetric(
            name="More than one duration metric", hour=2, minute=3, second=1.1
        )
        assert more_than_one_duration_metric.get_duration_string() == "2 hours 3 minutes 1.10 seconds"
        # Checking get_duration_string method for 0, 1, ">1" values specified as DurationMetric parameters
        mixed_conditions_metric = self.mixed_conditions_duration_metric()
        assert mixed_conditions_metric.get_duration_string() == "1 minute 2.10 seconds"

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        3. Check positive scenario when row_labels and column_labels parameters specified during MatrixMetric object
        initialization
        4. Check ValueError exception raised when row_labels parameter length is not equal to number of rows of
        MatrixMetric object
        5. Check ValueError exception raised when column_labels parameter length is not equal to number of columns of
        MatrixMetric object
        """
        matrix_metric = self.normalized_matrix_metric()

        required_normalised_matrix_data = np.array([[0, 0.5, 0.5], [0, 0.5, 0.5], [1, 0, 0]])
        assert np.array_equal(required_normalised_matrix_data, matrix_metric.matrix_values)
        assert repr(matrix_metric) == (
            "MatrixMetric(name=`test matrix`, matrix_values=(3x3) matrix, row labels=None, column labels=None)"
        )

        with warnings.catch_warnings():
            # there is a matrix with zero sum in row, so we expect 0/0 division.
            warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
            matrix_data_with_zero_sum = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0]])
            matrix_metric_with_zero_sum = MatrixMetric(
                name="test matrix",
                matrix_values=matrix_data_with_zero_sum,
                normalize=True,
            )

        required_normalised_matrix_data_with_zero_sum = np.array([[0, 0, 0], [0, 0.5, 0.5], [1, 0, 0]])
        assert np.array_equal(
            required_normalised_matrix_data_with_zero_sum,
            matrix_metric_with_zero_sum.matrix_values,
        )
        # Checking scenario with specified row_labels and column_labels parameters
        matrix_metric_with_labels_name = "MatrixMetric with row and column labels"
        row_labels = ["row_1", "row_2", "row_3"]
        column_labels = ["column_1", "column_2", "column_3"]
        matrix_metric_with_labels = MatrixMetric(
            name=matrix_metric_with_labels_name,
            matrix_values=self.matrix_data(),
            row_labels=row_labels,
            column_labels=column_labels,
        )
        assert matrix_metric_with_labels.name == matrix_metric_with_labels_name
        assert np.array_equal(self.matrix_data(), matrix_metric_with_labels.matrix_values)
        assert matrix_metric_with_labels.row_labels == row_labels
        assert matrix_metric_with_labels.column_labels == column_labels
        assert matrix_metric_with_labels.type() == "matrix"
        assert repr(matrix_metric_with_labels) == (
            "MatrixMetric(name=`MatrixMetric with row and column labels`, matrix_values=(3x3) matrix, "
            "row labels=['row_1', 'row_2', 'row_3'], column labels=['column_1', 'column_2', 'column_3'])"
        )
        # Checking ValueError exception raised when row_labels parameter length not equal to number of rows
        for incorrect_row_labels in (
            ["row_1", "row_2"],
            ["row_1", "row_2", "row_3", "row_4"],
        ):
            with pytest.raises(ValueError):
                MatrixMetric(
                    name="MatrixMetric with incorrect number of row labels",
                    matrix_values=self.matrix_data(),
                    row_labels=incorrect_row_labels,
                )
        # Checking ValueError exception raised when column_labels parameter length not equal to number of columns
        for incorrect_column_labels in (
            ["column_1", "column_2"],
            ["column_1", "column_2", "column_3", "column_4"],
        ):
            with pytest.raises(ValueError):
                MatrixMetric(
                    name="MatrixMetric with incorrect number of column labels",
                    matrix_values=self.matrix_data(),
                    column_labels=incorrect_column_labels,
                )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestCountMetric:
    @staticmethod
    def count_metric() -> CountMetric:
        return CountMetric(name="Test CountMetric", value=10)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_count_metric(self):
        """
        <b>Description:</b>
        Check CountMetric class

        <b>Input data:</b>
        CountMetric object with specified name and value parameters

        <b>Expected results:</b>
        Test passes if CountMetric object name, value and type attributes return expected values
        """
        for name, value in ["null metric", "positive metric"], [0, 10]:
            count_metric = CountMetric(name=name, value=value)
            assert count_metric.name == name
            assert count_metric.value == value
            assert count_metric.type() == "count"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestInfoMetric:
    @staticmethod
    def info_metric() -> InfoMetric:
        return InfoMetric(name="Test InfoMetric", value="This Metric is prepared for test purposes")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_info_metric(self):
        """
        <b>Description:</b>
        Check InfoMetric class

        <b>Input data:</b>
        InfoMetric object with specified name and value parameters

        <b>Expected results:</b>
        Test passes if InfoMetric object name, value and type attributes return expected values
        """
        info_metric = self.info_metric()
        assert info_metric.name == "Test InfoMetric"
        assert info_metric.value == "This Metric is prepared for test purposes"
        assert info_metric.type() == "string"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDateMetric:
    @staticmethod
    def date_metric_no_date_specified() -> DateMetric:
        return DateMetric(name="DateMetric with not specified date")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_date_metric(self):
        """
        <b>Description:</b>
        Check DateMetric class

        <b>Input data:</b>
        DateMetric object with specified name and date parameters

        <b>Expected results:</b>
        Test passes if DateMetric object name, date and type attributes return expected values

        <b>Steps</b>
        1. Check name, date and type attributes of DateMetric object with not specified date parameter
        2. Check name, date and type attributes of DateMetric object with specified date parameter
        """
        # Check for DateMetric with not specified date parameter
        date_not_specified_metric = self.date_metric_no_date_specified()
        assert date_not_specified_metric.name == "DateMetric with not specified date"
        assert isinstance(date_not_specified_metric.date, datetime.datetime)
        assert date_not_specified_metric.type() == "date"
        # Check for DateMetric with specified date parameter
        date_specified_metric_name = "DateMetric with specified date"
        date_expected = datetime.datetime(year=2020, month=11, day=29, hour=13, minute=25, second=10, microsecond=3)
        date_specified_metric = DateMetric(name=date_specified_metric_name, date=date_expected)
        assert date_specified_metric.name == date_specified_metric_name
        assert date_specified_metric.date == date_expected
        assert date_specified_metric.type() == "date"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestScoreMetric:
    @staticmethod
    def score_metric() -> ScoreMetric:
        return ScoreMetric(name="Test ScoreMetric", value=2.0)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_score_metric(self):
        """
        <b>Description:</b>
        Check ScoreMetric class

        <b>Input data:</b>
        ScoreMetric object with specified name and value parameters

        <b>Expected results:</b>
        Test passes if ScoreMetric object name, value and type attributes and __eq__ and __repr__ methods return
        expected values

        <b>Steps</b>
        1. Check name, value and type attributes for ScoreMetric object
        2. Check ValueError exception raised when ScoreMetric name parameter is float NaN
        3. Check __eq__ method for ScoreMetric object
        4. Check __repr__ method for ScoreMetric object
        """
        # Checking ScoreMetric object attributes
        score_metric = self.score_metric()
        assert score_metric.name == "Test ScoreMetric"
        assert score_metric.value == 2.0
        assert score_metric.type() == "score"
        # Checking exception raised when value is NaN
        with pytest.raises(ValueError):
            ScoreMetric(name="Test ScoreMetric", value=float("nan"))
        # Checking __eq__ method
        equal_score_metric = ScoreMetric(name="Test ScoreMetric", value=2.0)
        # Checking __eq__ method for equal ScoreMetric objects
        assert score_metric == equal_score_metric
        # Checking __eq__ method for ScoreMetric objects with unequal names
        different_name_score_metric = ScoreMetric(name="Other name ScoreMetric", value=2.0)
        assert score_metric != different_name_score_metric
        # Checking __eq__ method for ScoreMetric objects with unequal values
        different_value_score_metric = ScoreMetric(name="Test ScoreMetric", value=3.4)
        assert score_metric != different_value_score_metric
        # Checking __eq__ method by comparing ScoreMetric object with different type object
        assert score_metric != str
        # Checking __repr__ method
        assert repr(score_metric) == "ScoreMetric(name=`Test ScoreMetric`, score=`2.0`)"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestCurveMetric:
    @staticmethod
    def ys() -> list:
        return [2.0, 4.1, 3.3, 8.2, 7.1]

    def curve_metric(self) -> CurveMetric:
        xs = [0.0, 0.1, 0.2, 0.3, 0.4]
        return CurveMetric(name="Test CurveMetric", ys=self.ys(), xs=xs)

    def x_not_specified_curve_metric(self) -> CurveMetric:
        return CurveMetric(name="x not specified CurveMetric", ys=self.ys())

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_curve_metric(self):
        """
        <b>Description:</b>
        Check CurveMetric class

        <b>Input data:</b>
        CurveMetric object with specified name, xs and ys parameters

        <b>Expected results:</b>
        Test passes if CurveMetric object has expected attributes and methods

        <b>Steps</b>
        1. Check name, ys, xs and type attributes of CurveMetric object
        2. Check name, ys, xs and type attributes of CurveMetric object with not specified xs initialization parameter
        3. Check ValueError exception raised when length of ys CurveMetric object initialization parameter is not equal
        to length
        of xs parameter
        4. Check __repr__ method for CurveMetric object
        """
        # Checking positive scenario
        curve_metric = self.curve_metric()
        assert curve_metric.name == "Test CurveMetric"
        assert curve_metric.xs == [0.0, 0.1, 0.2, 0.3, 0.4]
        assert curve_metric.ys == self.ys()
        assert curve_metric.type() == "curve"
        # Checking positive scenario with not specified xs parameter
        x_not_specified_curve_metric = self.x_not_specified_curve_metric()
        assert x_not_specified_curve_metric.name == "x not specified CurveMetric"
        assert x_not_specified_curve_metric.xs == [1, 2, 3, 4, 5]
        assert x_not_specified_curve_metric.ys == self.ys()
        assert x_not_specified_curve_metric.type() == "curve"
        # Checking ValueError exception raised when len(ys) != len(xs)
        with pytest.raises(ValueError):
            CurveMetric(name="Negative CurveMetric Scenario", ys=[0.0, 0.1], xs=[1, 2, 3])
        # Checking __repr__ method
        assert repr(curve_metric) == "CurveMetric(name=`Test CurveMetric`, ys=(5 values), xs=(5 values))"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestNullMetric:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_null_metric(self):
        """
        <b>Description:</b>
        Check NullMetric class

        <b>Input data:</b>
        NullMetric object

        <b>Expected results:</b>
        Test passes if NullMetric object name and type attributes and __repr__ and __eq__ methods return expected values
        """
        # Checking NullMetric attributes
        null_metric = NullMetric()
        assert null_metric.name == "NullMetric"
        assert null_metric.type() == "null"
        # Checking NullMetric __repr__ method
        assert repr(null_metric) == "NullMetric()"
        # Checking NullMetric __eq__ method
        equal_null_metric = NullMetric()
        assert null_metric == equal_null_metric
        # Checking __eq__ method by comparing NullMetric object with different type object
        assert null_metric != TestInfoMetric().info_metric()


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestVisualizationType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_visualization_type(self):
        """
        <b>Description:</b>
        Check VisualizationType Enum class elements

        <b>Expected results:</b>
        Test passes if VisualizationType Enum class length equal expected value and its elements have expected
        sequence numbers
        """
        assert len(VisualizationType) == 5
        assert VisualizationType.TEXT.value == 0
        assert VisualizationType.RADIAL_BAR.value == 1
        assert VisualizationType.BAR.value == 2
        assert VisualizationType.LINE.value == 3
        assert VisualizationType.MATRIX.value == 4


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestColorPalette:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_color_palette(self):
        """
        <b>Description:</b>
        Check ColorPalette Enum class elements

        <b>Expected results:</b>
        Test passes if ColorPalette Enum class length equal expected value and its elements have expected
        sequence numbers
        """
        assert len(ColorPalette) == 2
        assert ColorPalette.DEFAULT.value == 0
        assert ColorPalette.LABEL.value == 1


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestVisualizationInfo:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_visualization_info(self):
        """
        <b>Description:</b>
        Check VisualizationInfo class

        <b>Input data:</b>
        VisualizationInfo object with specified name, visualisation_type and palette parameters

        <b>Expected results:</b>
        Test passes if VisualizationInfo object name and palette parameters and type and __repr__ methods return
        expected values

        <b>Steps</b>
        1. Check name, type and palette attributes and __repr__ method for VisualizationInfo object with not specified
        palette parameter
        2. Check name, type and palette attributes and __repr__ method for VisualizationInfo object with specified
        palette parameter
        """
        # Checks for not specified palette parameter
        no_palette_specified_name = "No palette specified VisualizationInfo"
        for visualisation_type in VisualizationType:
            visualisation_info = VisualizationInfo(
                name=no_palette_specified_name, visualisation_type=visualisation_type
            )
            assert visualisation_info.name == no_palette_specified_name
            assert visualisation_info.type == visualisation_type
            assert visualisation_info.palette == ColorPalette.DEFAULT
            assert repr(visualisation_info) == (
                f"VisualizationInfo(name='No palette specified VisualizationInfo', "
                f"type='{visualisation_type.name}', palette='DEFAULT')"
            )
        # Check for specified palette parameter
        palette_specified_name = "VisualizationInfo with palette parameter set to LABEL"
        for visualisation_type in VisualizationType:
            visualisation_info = VisualizationInfo(
                name=palette_specified_name,
                visualisation_type=visualisation_type,
                palette=ColorPalette.LABEL,
            )
            assert visualisation_info.name == palette_specified_name
            assert visualisation_info.type == visualisation_type
            assert visualisation_info.palette == ColorPalette.LABEL
            assert repr(visualisation_info) == (
                f"VisualizationInfo(name='VisualizationInfo with palette parameter set "
                f"to LABEL', type='{visualisation_type.name}', palette='LABEL')"
            )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTextChartInfo:
    @staticmethod
    def text_chart_info():
        return TextChartInfo("Test TextChartInfo")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_text_chart_info(self):
        """
        <b>Description:</b>
        Check TextChartInfo class

        <b>Input data:</b>
        TextChartInfo object with specified name parameter

        <b>Expected results:</b>
        Test passes if TextChartInfo object name and type attributes and __repr__ method return expected values
        """
        text_chart_info = self.text_chart_info()
        assert text_chart_info.name == "Test TextChartInfo"
        assert text_chart_info.type == VisualizationType.TEXT
        assert repr(text_chart_info) == ("TextChartInfo(name='Test TextChartInfo, " "'type='VisualizationType.TEXT')")


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLineChartInfo:
    @staticmethod
    def default_parameters_line_chart_info():
        return LineChartInfo("Test default parameters LineChartInfo")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_line_chart_info(self):
        """
        <b>Description:</b>
        Check LineChartInfo class

        <b>Input data:</b>
        LineChartInfo object with specified name, x_axis_label, y_axis_label and palette parameters

        <b>Expected results:</b>
        Test passes if LineChartInfo object name, x_axis_label, y_axis_label, palette and type attributes and
        __repr__ method return expected values

        <b>Steps</b>
        1. Check name, x_axis_label, y_axis_label, palette and type attributes and __repr__ method values for
        LineChartInfo with specified x_axis_label, y_axis_label and palette parameters
        2. Check name, x_axis_label, y_axis_label, palette and type attributes and __repr__ method values for
        LineChartInfo with not specified x_axis_label, y_axis_label and palette parameters
        """
        # Scenario for specified parameters
        line_chart_info_name = "Test LineChartInfo"
        x_axis_label = "Test x-axis label for LineChartInfo"
        y_axis_label = "Test y-axis label for LineChartInfo"
        palette = ColorPalette.LABEL
        parameters_specified_line_chart_info = LineChartInfo(
            name=line_chart_info_name,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            palette=palette,
        )
        assert parameters_specified_line_chart_info.name == line_chart_info_name
        assert parameters_specified_line_chart_info.x_axis_label == x_axis_label
        assert parameters_specified_line_chart_info.y_axis_label == y_axis_label
        assert parameters_specified_line_chart_info.palette == palette
        assert parameters_specified_line_chart_info.type == VisualizationType.LINE
        assert repr(parameters_specified_line_chart_info) == (
            "LineChartInfo(name='Test LineChartInfo, 'type='VisualizationType.LINE', x_axis_label='Test x-axis "
            "label for LineChartInfo', y_axis_label='Test y-axis label for LineChartInfo')"
        )
        # Scenario for default parameters
        default_values_line_chart_info = self.default_parameters_line_chart_info()
        assert default_values_line_chart_info.name == "Test default parameters LineChartInfo"
        assert default_values_line_chart_info.x_axis_label == ""
        assert default_values_line_chart_info.y_axis_label == ""
        assert default_values_line_chart_info.palette == ColorPalette.DEFAULT
        assert default_values_line_chart_info.type == VisualizationType.LINE
        assert repr(default_values_line_chart_info) == (
            "LineChartInfo(name='Test default parameters LineChartInfo, 'type='VisualizationType.LINE', "
            "x_axis_label='', y_axis_label='')"
        )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestBarChartInfo:
    @staticmethod
    def default_parameters_bar_chart_info():
        return BarChartInfo(name="BarChartInfo with default parameters")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_bar_chart_info(self):
        """
        <b>Description:</b>
        Check BarChartInfo class

        <b>Input data:</b>
        BarChartInfo object with specified name, palette and visualization_type parameters

        <b>Expected results:</b>
        Test passes if BarChartInfo object name, type and palette attributes and __repr__ method return expected values

        <b>Steps</b>
        1. Check name, palette and type attributes and __repr__ method values for BarChartInfo object with specified
        palette and visualisation_type parameters
        2. Check name, palette and type attributes and __repr__ method values for BarChartInfo object with not
        specified palette and visualisation_type parameters
        3. Check ValueError exception raised when visualization_type BarChartInfo object initialization parameter is
        not equal to BAR or RADIAL_BAR
        """
        # Scenario for specified parameters
        bar_chart_info_name = "Test BarChartInfo"
        for visualisation_type in [VisualizationType.BAR, VisualizationType.RADIAL_BAR]:
            bar_chart_info = BarChartInfo(
                name=bar_chart_info_name,
                palette=ColorPalette.LABEL,
                visualization_type=visualisation_type,
            )
            assert bar_chart_info.name == bar_chart_info_name
            assert bar_chart_info.palette == ColorPalette.LABEL
            assert bar_chart_info.type == visualisation_type
            assert repr(bar_chart_info) == (f"BarChartInfo(name='Test BarChartInfo', " f"type='{visualisation_type}')")
        # Scenario for default parameters
        default_values_bar_chart_info = self.default_parameters_bar_chart_info()
        assert default_values_bar_chart_info.name == "BarChartInfo with default parameters"
        assert default_values_bar_chart_info.palette == ColorPalette.DEFAULT
        assert default_values_bar_chart_info.type == VisualizationType.BAR
        assert repr(default_values_bar_chart_info) == (
            "BarChartInfo(name='BarChartInfo with default parameters', " "type='VisualizationType.BAR')"
        )
        # Check ValueError exception raised when visualization_type not equal to BAR or RADIAL_BAR
        for visualisation_type in [
            VisualizationType.TEXT,
            VisualizationType.LINE,
            VisualizationType.MATRIX,
        ]:
            with pytest.raises(ValueError):
                BarChartInfo(name=bar_chart_info_name, visualization_type=visualisation_type)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestMatrixChartInfo:
    @staticmethod
    def default_values_matrix_chart_info():
        return MatrixChartInfo("Test MatrixCharInfo with default parameters")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_matrix_chart_info(self):
        """
        <b>Description:</b>
        Check MatrixChartInfo class

        <b>Input data:</b>
        MatrixChartInfo object with specified name, header, row_header, column_header and palette parameters

        <b>Expected results:</b>
        Test passes if MatrixChartInfo object name, header, row_header, column_header, palette and type attributes and
        __repr__ method return expected values

        <b>Steps</b>
        1. Check name, header, row_header, column_header, palette and type attributes and __repr__ method values for
        MatrixChartInfo object with specified parameters
        2. Check name, header, row_header, column_header, palette and type attributes and __repr__ method values for
        MatrixChartInfo object with default parameters
        """
        # Check for specified parameters
        matrix_chart_info_name = "Test MatrixChartInfo"
        matrix_chart_info_header = "Header of Test MatrixChartInfo"
        matrix_chart_info_row_header = "Specified row header"
        matrix_chart_info_column_header = "Specified column header"
        matrix_chart_info = MatrixChartInfo(
            name=matrix_chart_info_name,
            header=matrix_chart_info_header,
            row_header=matrix_chart_info_row_header,
            column_header=matrix_chart_info_column_header,
            palette=ColorPalette.LABEL,
        )
        assert matrix_chart_info.name == matrix_chart_info_name
        assert matrix_chart_info.header == matrix_chart_info_header
        assert matrix_chart_info.row_header == matrix_chart_info_row_header
        assert matrix_chart_info.column_header == matrix_chart_info_column_header
        assert matrix_chart_info.palette == ColorPalette.LABEL
        assert matrix_chart_info.type == VisualizationType.MATRIX
        assert repr(matrix_chart_info) == (
            "MatrixChartInfo(name='Test MatrixChartInfo', type='VisualizationType.MATRIX', header='Header of Test "
            "MatrixChartInfo', row_header='Specified row header', column_header='Specified column header')"
        )
        # Check for default parameters
        default_parameters_matrix_chart_info = self.default_values_matrix_chart_info()
        assert default_parameters_matrix_chart_info.name == "Test MatrixCharInfo with default parameters"
        assert default_parameters_matrix_chart_info.palette == ColorPalette.DEFAULT
        assert default_parameters_matrix_chart_info.type == VisualizationType.MATRIX
        with pytest.raises(AttributeError):
            default_parameters_matrix_chart_info.header
        with pytest.raises(AttributeError):
            default_parameters_matrix_chart_info.row_header
        with pytest.raises(AttributeError):
            default_parameters_matrix_chart_info.column_header
        with pytest.raises(AttributeError):
            repr(default_parameters_matrix_chart_info)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestMatrixMetricsGroup:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_matrix_metrics_group(self):
        """
        <b>Description:</b>
        Check MatrixMetricsGroup class

        <b>Input data:</b>
        MatrixMetricsGroup object with specified metrics and visualization_info parameters

        <b>Expected results:</b>
        Test passes if MatrixMetricsGroup object metrics and visualization_info attributes return expected values

        <b>Steps</b>
        1. Check metrics and visualization_info attributes for MatrixMetricsGroup object with specified parameters
        2. Check ValueError raised when MatrixMetricsGroup object has metrics parameter equal to None or empty list
        3. Check ValueError raised when MatrixMetricsGroup object has visualization_info parameter equal to None
        """
        # Positive scenario for MatrixMetricsGroup object with specified parameters
        with warnings.catch_warnings():
            # there is a matrix with zero sum in row, so we expect 0/0 division.
            warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
            matrix_metrics = [
                TestMetrics().normalized_matrix_metric(),
                TestMetrics().normalized_matrix_zero_sum(),
            ]
        matrix_chart_info = TestMatrixChartInfo.default_values_matrix_chart_info()
        matrix_metrics_group = MatrixMetricsGroup(metrics=matrix_metrics, visualization_info=matrix_chart_info)
        assert matrix_metrics_group.metrics == matrix_metrics
        assert matrix_metrics_group.visualization_info == matrix_chart_info
        # Negative scenarios for MatrixMetricsGroup object with metrics parameter equal to None or []
        for incorrect_metrics in [None, []]:
            with pytest.raises(ValueError):
                MatrixMetricsGroup(metrics=incorrect_metrics, visualization_info=matrix_chart_info)
        # Negative scenario for MatrixMetricsGroup object with visualization_info parameter equal to None
        with pytest.raises(ValueError):
            MatrixMetricsGroup(metrics=matrix_metrics, visualization_info=None)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLineMetricsGroup:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_line_metrics_group(self):
        """
        <b>Description:</b>
        Check LineMetricsGroup class

        <b>Input data:</b>
        LineMetricsGroup object with specified metrics and visualization_info parameters

        <b>Expected results:</b>
        Test passes if LineMetricsGroup object metrics and visualization_info attributes return expected values

        <b>Steps</b>
        1. Check metrics and visualization_info attributes for LineMetricsGroup object with specified parameters
        2. Check ValueError raised when LineMetricsGroup object has metrics parameter equal to None or empty tuple
        3. Check ValueError raised when LineMetricsGroup object has visualization_info parameter equal to None
        """
        # Positive scenario for TestLineMetricsGroup object with specified parameters
        curve_metrics = (
            TestCurveMetric().curve_metric(),
            TestCurveMetric().x_not_specified_curve_metric(),
        )
        line_chart_info = TestLineChartInfo().default_parameters_line_chart_info()
        line_metrics_group = LineMetricsGroup(metrics=curve_metrics, visualization_info=line_chart_info)
        assert line_metrics_group.metrics == curve_metrics
        assert line_metrics_group.visualization_info == line_chart_info
        # Negative scenarios for LineMetricsGroup object with metrics parameter equal to None or []
        for incorrect_metrics in [None, ()]:
            with pytest.raises(ValueError):
                LineMetricsGroup(metrics=incorrect_metrics, visualization_info=line_chart_info)
        # Negative scenario for LineMetricsGroup object with visualization_info parameter equal to None
        with pytest.raises(ValueError):
            LineMetricsGroup(metrics=curve_metrics, visualization_info=None)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestBarMetricsGroup:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_bar_metrics_group(self):
        """
        <b>Description:</b>
        Check BarMetricsGroup class

        <b>Input data:</b>
        BarMetricsGroup object with specified metrics and visualization_info parameters

        <b>Expected results:</b>
        Test passes if BarMetricsGroup object metrics and visualization_info attributes return expected values

        <b>Steps</b>
        1. Check metrics and visualization_info attributes for BarMetricsGroup object with specified parameters
        2. Check ValueError raised when BarMetricsGroup object has metrics parameter equal to None or empty list
        3. Check ValueError raised when BarMetricsGroup object has visualization_info parameter equal to None
        """
        # Positive scenario for BarMetricsGroup object with specified parameters
        bar_metrics = [
            TestScoreMetric().score_metric(),
            TestCountMetric().count_metric(),
        ]
        bar_chart_info = TestBarChartInfo().default_parameters_bar_chart_info()
        bar_metrics_group = BarMetricsGroup(metrics=bar_metrics, visualization_info=bar_chart_info)
        assert bar_metrics_group.metrics == bar_metrics
        assert bar_metrics_group.visualization_info == bar_chart_info
        # Negative scenarios for BarMetricsGroup object with metrics parameter equal to None or []
        for incorrect_metrics in [None, []]:
            with pytest.raises(ValueError):
                BarMetricsGroup(metrics=incorrect_metrics, visualization_info=bar_chart_info)
        # Negative scenario for BarMetricsGroup object with visualization_info parameter equal to None
        with pytest.raises(ValueError):
            BarMetricsGroup(metrics=bar_metrics, visualization_info=None)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTextMetricsGroup:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_text_metrics_group(self):
        """
        <b>Description:</b>
        Check TextMetricsGroup class

        <b>Input data:</b>
        TextMetricsGroup object with specified metrics and visualization_info parameters

        <b>Expected results:</b>
        Test passes if TextMetricsGroup object metrics and visualization_info attributes return expected values

        <b>Steps</b>
        1. Check metrics and visualization_info attributes for TextMetricsGroup object with specified parameters
        2. Check ValueError raised when TextMetricsGroup object has metrics parameter length more than 1
        3. Check ValueError raised when TextMetricsGroup object has metrics parameter equal empty tuple
        4. Check ValueError raised when TextMetricsGroup object has visualization_info parameter equal to None
        """
        # Positive scenario for TextMetricsGroup object with specified parameters
        score_metric = TestScoreMetric().score_metric()
        count_metric = TestCountMetric().count_metric()
        text_chart_info = TestTextChartInfo().text_chart_info()
        for metric in [
            score_metric,
            count_metric,
            TestInfoMetric().info_metric(),
            TestDateMetric().date_metric_no_date_specified(),
            TestMetrics().mixed_conditions_duration_metric(),
        ]:
            text_metric_group = TextMetricsGroup(metrics=[metric], visualization_info=text_chart_info)
            assert text_metric_group.metrics == [metric]
            assert text_metric_group.visualization_info == text_chart_info
        # Negative scenarios for TextMetricsGroup object with metrics parameter length equal to 2
        with pytest.raises(ValueError):
            TextMetricsGroup(metrics=(score_metric, count_metric), visualization_info=text_chart_info)
        # Negative scenarios for TextMetricsGroup object with metrics parameter equal ()
        with pytest.raises(ValueError):
            TextMetricsGroup(metrics=(), visualization_info=text_chart_info)
        # Negative scenario for TextMetricsGroup object with visualization_info parameter equal to None
        with pytest.raises(ValueError):
            TextMetricsGroup(metrics=[score_metric], visualization_info=None)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestPerformance:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_performance(self):
        """
        <b>Description:</b>
        Check Performance class

        <b>Input data:</b>
        Performance object with specified score and dashboard_metrics parameters

        <b>Expected results:</b>
        Test passes if Performance object score and dashboard_metrics attributes and __eq__ and __repr__ method return
        expected values

        <b>Steps</b>
        1. Check score and dashboard_metrics attributes for Performance object with not specified dashboard_metrics
        parameter
        2. Check score and dashboard_metrics attributes for Performance object with specified dashboard_metrics
        parameter
        3. Check __eq__ method for equal Performance objects, Performance objects with different dashboard_metrics
        attributes, Performance objects with different score attributes
        4. Check __repr__ method
        5. Check ValueError exception raised when score attributes type not equal to ScoreMetric
        """
        # Positive scenario for Performance object with default parameters
        score_metric = TestScoreMetric().score_metric()
        default_parameters_performance = Performance(score_metric)
        assert default_parameters_performance.score == score_metric
        assert default_parameters_performance.dashboard_metrics == []
        # Positive scenario for Performance object with specified dashboard_metrics  parameter
        # Preparing dashboard metrics list
        with warnings.catch_warnings():
            # there is a matrix with zero sum in row, so we expect 0/0 division.
            warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
            matrix_metrics = [
                TestMetrics().normalized_matrix_metric(),
                TestMetrics().normalized_matrix_zero_sum(),
            ]
        matrix_chart_info = TestMatrixChartInfo.default_values_matrix_chart_info()
        matrix_metrics_group = MatrixMetricsGroup(metrics=matrix_metrics, visualization_info=matrix_chart_info)
        curve_metrics = (
            TestCurveMetric().curve_metric(),
            TestCurveMetric().x_not_specified_curve_metric(),
        )
        line_chart_info = TestLineChartInfo().default_parameters_line_chart_info()
        line_metrics_group = LineMetricsGroup(metrics=curve_metrics, visualization_info=line_chart_info)
        bar_metrics = [
            TestScoreMetric().score_metric(),
            TestCountMetric().count_metric(),
        ]
        bar_chart_info = TestBarChartInfo().default_parameters_bar_chart_info()
        bar_metrics_group = BarMetricsGroup(metrics=bar_metrics, visualization_info=bar_chart_info)
        text_score_metric = TestScoreMetric().score_metric()
        text_chart_info = TestTextChartInfo().text_chart_info()
        text_metric_group = TextMetricsGroup(metrics=[text_score_metric], visualization_info=text_chart_info)
        dashboard_metrics = [
            matrix_metrics_group,
            line_metrics_group,
            bar_metrics_group,
            text_metric_group,
        ]
        # Checking Performance attributes
        specified_parameters_performance = Performance(score=score_metric, dashboard_metrics=dashboard_metrics)
        assert specified_parameters_performance.score == score_metric
        assert specified_parameters_performance.dashboard_metrics == dashboard_metrics
        # Checking __eq__ method
        equal_default_parameters_performance = Performance(score_metric)
        assert default_parameters_performance == equal_default_parameters_performance
        different_metrics_performance = Performance(score_metric, [matrix_metrics_group])
        assert default_parameters_performance == different_metrics_performance
        unequal_score_metric = ScoreMetric(name="Unequal ScoreMetric", value=1.0)
        assert default_parameters_performance != Performance(unequal_score_metric)
        assert default_parameters_performance != str
        # Checking __repr__ method
        assert repr(default_parameters_performance) == "Performance(score: 2.0, dashboard: (0 metric groups))"
        assert repr(specified_parameters_performance) == "Performance(score: 2.0, dashboard: (4 metric groups))"
        # Checking ValueError exception raised when score parameter not ScoreMetric class
        count_metric = TestCountMetric().count_metric()
        with pytest.raises(ValueError):
            Performance(count_metric)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestNullPerformance:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_null_performance(self):
        """
        <b>Description:</b>
        Check NullPerformance class

        <b>Input data:</b>
        NullPerformance object

        <b>Expected results:</b>
        Test passes if NullPerformance object score and dashboard_metrics attributes and __repr__ and __eq__ methods
        return expected values

        <b>Steps</b>
        1. Check NullPerformance object score and dashboard_metrics attributes
        2. Check NullPerformance object __repr__ method
        3. Check NullPerformance object __eq__ method
        """
        # Checking NullPerformance score and dashboard_metrics attributes
        null_performance = NullPerformance()
        assert null_performance.score == ScoreMetric(name="Null score", value=0.0)
        assert null_performance.dashboard_metrics == []
        # Checking NullPerformance __repr__ method
        assert repr(null_performance) == "NullPerformance()"
        # Checking __eq__ method for equal NullPerformance objects
        equal_null_performance = NullPerformance()
        assert null_performance == equal_null_performance
        # Checking NullPerformance __eq__ method by comparing with Performance object
        score_metric = TestScoreMetric().score_metric()
        performance = Performance(score_metric)
        assert null_performance != performance


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestMultiScorePerformance:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_multi_score_performance(self):
        """
        <b>Description:</b>
        Check MultiScorePerformance class

        <b>Input data:</b>
        MultiScorePerformance object with specified score

        <b>Expected results:</b>
        Test passes if MultiScorePerformance object score attribute and __eq__ and __repr__ method return
        expected values

        <b>Steps</b>
        1. Check primary and additional score attributes for MultiScorePerformance object
        2. Check primary and additional score attributes for MultiScorePerformance object when only primary score is
        passed
        3. Check primary and additional score attributes for MultiScorePerformance object when only additional score is
        passed
        4. Check __eq__ method for equal and unequal Performance objects
        5. Check __repr__ method
        """
        # Positive scenario for Performance object with default parameters
        primary_score = TestScoreMetric().score_metric()
        additional_score = TestScoreMetric().score_metric()
        default_parameters_performance = MultiScorePerformance(primary_score, [additional_score])
        assert default_parameters_performance.score == primary_score
        assert default_parameters_performance.primary_score == primary_score
        assert default_parameters_performance.additional_scores == [additional_score]
        assert default_parameters_performance.dashboard_metrics == []
        # Positive scenario for Performance object with only primary metric
        only_primary_performance = MultiScorePerformance(primary_score)
        assert only_primary_performance.score == primary_score
        assert only_primary_performance.primary_score == primary_score
        assert only_primary_performance.additional_scores == []
        assert only_primary_performance.dashboard_metrics == []
        # Positive scenario for Performance object with only additional metric
        only_additional_performance = MultiScorePerformance(additional_scores=[additional_score])
        assert only_additional_performance.score == additional_score
        assert only_additional_performance.primary_score is None
        assert only_additional_performance.additional_scores == [additional_score]
        assert only_additional_performance.dashboard_metrics == []
        # Checking __eq__ method
        equal_default_parameters_performance = MultiScorePerformance(primary_score, [additional_score])
        assert default_parameters_performance == equal_default_parameters_performance
        assert default_parameters_performance != only_primary_performance
        # Checking __repr__ method
        assert (
            repr(default_parameters_performance)
            == "MultiScorePerformance(score: 2.0, primary_metric: ScoreMetric(name=`Test ScoreMetric`, score=`2.0`), "
            "additional_metrics: (1 metrics), dashboard: (0 metric groups))"
        )
