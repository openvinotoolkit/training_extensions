"""This module implements the Metric entities."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
import datetime
import logging
import math
from enum import Enum
from typing import Generic, List, Optional, Sequence, TypeVar, Union

import numpy as np

from otx.api.utils.time_utils import now


class MetricEntity(metaclass=abc.ABCMeta):
    """This interface represents a metric, which is the smallest building block for the performance statistics.

    It only contains the name of the metric.
    See also :class:`MetricsGroup` and :class:`Performance` for the structure of performance statistics.
    """

    __name = None

    @property
    def name(self):
        """Returns the name of the Metric Entity."""
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    @staticmethod
    def type() -> str:
        """Returns the type of the MetricEntity, e.g. "curve."""


class CountMetric(MetricEntity):
    """This metric represents an integer value.

    Args:
        name: The name of the metric
        value: The value of the metric

    Example:
        The count for number of images in a project

    >>> count_metric = CountMetric(name="Number of images", value=20)

    """

    value: int

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "count"


class InfoMetric(MetricEntity):
    """This metric represents a string value.

    Args:
        name: The name of the info metric
        value: The info of the metric

    Example:
        An info metric of training from scratch

    >>> info_metric = InfoMetric(name="Model info", value="This model is trained from scratch")

    """

    value: str

    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "string"


class DateMetric(MetricEntity):
    """This metric represents a date time value.

    Args:
        name: The name of the date metric
        date: The datetime value of the metric

    Example:
        A DateMetric for model creation date (e.g., now).

    >>> metric = DateMetric(name="Model creation", date=datetime.datetime.now(datetime.timezone.utc))

    """

    date: datetime.datetime

    def __init__(self, name: str, date: Optional[datetime.datetime] = None):
        self.name = name
        if date is None:
            date = now()
        self.date = date

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "date"


class ScoreMetric(MetricEntity):
    """This metric represents a float value.

    This metric is typically used for storing performance metrics, such as accuracy, f-measure, dice score, etc.

    Args:
        name: The name of the score
        value: The value of the score

    Example:
        Accuracy of a model

    >>> score_metric = ScoreMetric(name="Model accuracy", value=0.5)

    """

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

        if math.isnan(value):
            raise ValueError("The value of a ScoreMetric is not allowed to be NaN.")

    def __eq__(self, other: object) -> bool:
        """Returns True if the score metrics are equal."""
        if not isinstance(other, ScoreMetric):
            return False
        return self.name == other.name and self.value == other.value

    def __repr__(self):
        """Returns the representation of the score metric."""
        return f"ScoreMetric(name=`{self.name}`, score=`{self.value}`)"

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "score"


class DurationMetric(MetricEntity):
    """This metric represents a duration metric, which include hour (int), minute (int), and second (float).

    Args:
        name: The name of the duration metric
        hour: The hour value of the metric
        minute: The minute value of the metric
        second: The second value of the metric

    Example:
        Creating a metric for training duration of 1 hour 5 minutes.

    >>> duration_metric = DurationMetric(name="Training duration", hour=1, minute=5, second=0)

    """

    def __init__(self, name: str, hour: int, minute: int, second: float):
        self.hour = hour
        self.minute = minute
        self.second = second
        self.name = name

    def get_duration_string(self) -> str:
        """Returns the string representation of the duration.

        Example:
            Duration string of 1 hour 1 minute and 1.50 seconds.

        >>> from otx.api.entities.metrics import DurationMetric
        >>> dur_met = DurationMetric("test", 1, 1, 1.5)  # 1 hour 1 minute and 1.5 seconds
        >>> dur_met.get_duration_string()
        '1 hour 1 minute 1.50 seconds'

        Returns:
            the string representation of the duration.
        """
        output: str = ""
        if self.hour != 0:
            output += f"{self.hour} hour{'s ' if self.hour > 1 else ' '}"
        if self.minute != 0:
            output += f"{self.minute} minute{'s ' if self.minute > 1 else ' '}"
        if self.second != 0:
            output += f"{self.second:.02f} second{'s ' if self.second > 1 else ' '}"
        output = output.strip()
        return output

    @staticmethod
    def from_seconds(name: str, seconds: float) -> "DurationMetric":
        """Returns a duration metrics, with name and converted durations from seconds.

        Example:
            Converting 70 seconds to duration metric.

        >>> from otx.api.entities.metrics import DurationMetric
        >>> dur_met = DurationMetric.from_seconds("test", 70)  # 1 hour 1 minute and 1.5 seconds
        >>> dur_met.get_duration_string()
        '1 minute 10.00 seconds'

        Args:
            name
            seconds

        Returns:
            DurationMetric: the duration metric with name and converted durations from seconds.
        """
        hour = int(seconds // 3600)
        modulo = seconds % 3600
        minute = int(modulo // 60)
        second = modulo % 60
        return DurationMetric(name=name, hour=hour, minute=minute, second=second)

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "duration"


class CurveMetric(MetricEntity):
    """This metric represents a curve. The coordinates are represented as x and y lists.

    Example:
        A line curve of: [(0,1), (1, 5), (2, 8)]

    >>> CurveMetric("Line", xs=[0, 1, 2], ys=[1, 5, 8])
    CurveMetric(name=`Line`, ys=(3 values), xs=(3 values))

    A curve can also be defined only using the y values. For example, a loss curve of loss values: [0.5, 0.2, 0.1].
    The x values will be automatically generated as a 1-based index (1, 2, 3, ...)

    >>> CurveMetric("Loss", ys=[0.5, 0.2, 0.1])
    CurveMetric(name=`Loss`, ys=(3 values), xs=(None values))

    Args:
        name: The name of the curve
        xs: the list of floats in x-axis
        ys: the list of floats in y-axis
    """

    def __init__(self, name: str, ys: List[float], xs: Optional[List[float]] = None):
        self.name = name
        self.__ys = ys
        if xs is not None:
            if len(xs) != len(self.__ys):
                raise ValueError(f"Curve error must contain the same length for x and y: ({len(xs)} vs {len(self.ys)})")
            self.__xs = xs
        else:
            # if x values are not provided, set them to the 1-index of the y values
            self.__xs = list(range(1, len(self.__ys) + 1))

    @property
    def ys(self) -> List[float]:
        """Returns the list of floats on y-axis."""
        return self.__ys

    @property
    def xs(self) -> List[float]:
        """Returns the list of floats on x-axis."""
        return self.__xs

    def __repr__(self):
        """Returns the string representation of the object."""
        return (
            f"CurveMetric(name=`{self.name}`, ys=({len(self.ys)} values), "
            f"xs=({len(self.xs) if self.xs is not None else 'None'} values))"
        )

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "curve"


class MatrixMetric(MetricEntity):
    """This metric represents a matrix. The cells are represented as a list of lists of integers.

    In the case of a confusion matrix, the rows represent the ground truth items and the columns represent the
    predicted items.

    Example:
        A matrix of: [[4,0,1], [0,3,2], [1,2,2]]

        >>> MatrixMetric("Confusion Matrix", matrix_values=np.array([[4,0,1], [0,3,2], [1,2,2]]))
        MatrixMetric(name=`Confusion Matrix`, matrix_values=(3x3) matrix, row labels=None, column labels=None)

    Args:
        name: The name of the matrix
        matrix_values: the matrix data
        row_labels: labels for the rows
        column_labels: labels for the columns
        normalize: set to True to normalize each row of the matrix
    """

    __row_labels: Optional[List[str]] = None
    __column_labels: Optional[List[str]] = None

    # pylint: disable=too-many-arguments; Requires refactor
    def __init__(
        self,
        name: str,
        matrix_values: np.ndarray,
        row_labels: Optional[List[str]] = None,
        column_labels: Optional[List[str]] = None,
        normalize: bool = False,
    ):
        self.name = name
        self.__matrix_values = matrix_values
        self.__matrix_values.astype(np.float32)

        if row_labels is not None:
            self.__row_labels = row_labels
            if self.__matrix_values.shape[0] != len(self.__row_labels):
                raise ValueError(
                    f"Number of rows of the matrix and number of row labels must be equal. The shape "
                    f"has {self.__matrix_values.shape[0]} rows and {len(self.__row_labels)} row labels"
                )

        if column_labels is not None:
            self.__column_labels = column_labels
            if self.__matrix_values.shape[1] != len(self.__column_labels):
                raise ValueError(
                    f"Number of columns of the matrix and number of column labels must be equal. The "
                    f"shape has {self.__matrix_values.shape[1]} columns and {len(self.__column_labels)} column "
                    "labels"
                )

        if normalize:
            self.normalize()

    @property
    def matrix_values(self) -> np.ndarray:
        """Returns the matrix data."""
        return self.__matrix_values

    @property
    def row_labels(self) -> Optional[List[str]]:
        """Returns the row labels."""
        return self.__row_labels

    @property
    def column_labels(self) -> Optional[List[str]]:
        """Returns the column labels."""
        return self.__column_labels

    def normalize(self):
        """Normalizes the confusion matrix by dividing by the sum of the rows."""
        self.__matrix_values = self.__matrix_values.astype(np.float32) / self.__matrix_values.astype(np.float32).sum(
            axis=1, keepdims=True
        )  # Divide all values by the sum of its row

        if not np.all(self.__matrix_values.sum(axis=1, keepdims=True) > 0):
            self.__matrix_values = np.nan_to_num(self.__matrix_values)

            logger = logging.getLogger(__name__)
            logger.warning("Replacing NaN in the matrix with zeroes since the sum of one (or more) row(s) was zero.")

    def __repr__(self):
        """Returns the string representation of the object."""
        return (
            f"MatrixMetric(name=`{self.name}`, matrix_values=({self.__matrix_values.shape[0]}x"
            f"{self.__matrix_values.shape[1]}) matrix, row labels={self.__row_labels}, column labels"
            f"={self.__column_labels})"
        )

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "matrix"


class NullMetric(MetricEntity):
    """Represents 'Metric not found'."""

    def __init__(self) -> None:
        self.name = "NullMetric"

    def __repr__(self):
        """Returns the string representation of the object."""
        return "NullMetric()"

    def __eq__(self, other):
        """Returns True if the other object is a NullMetric."""
        return isinstance(other, NullMetric)

    @staticmethod
    def type():
        """Returns the type of the MetricEntity."""
        return "null"


class VisualizationType(Enum):
    """This enum defines how the metrics will be visualized on the UI."""

    TEXT = 0
    RADIAL_BAR = 1
    BAR = 2
    LINE = 3
    MATRIX = 4


class ColorPalette(Enum):
    """Enum class specifying the color palette to be used by the UI to display statistics.

    If the statistics are per label, set to LABEL so the UI will use the label color palette.
    Otherwise, set to DEFAULT (allow the UI to choose a color palette)
    """

    DEFAULT = 0
    LABEL = 1


class VisualizationInfo:
    """This represents the visualization info a metrics group. See :class:`MetricsGroup`."""

    __type: VisualizationType
    name: str  # todo: this should be a part of MetricsGroup, not the visualization info.

    def __init__(
        self,
        name: str,
        visualisation_type: VisualizationType,
        palette: ColorPalette = ColorPalette.DEFAULT,
    ):
        self.__type = visualisation_type
        self.name = name
        self.palette = palette

    @property
    def type(self) -> VisualizationType:
        """Returns the type of the visualization."""
        return self.__type

    def __repr__(self):
        """Returns the string representation of the object."""
        return f"VisualizationInfo(name='{self.name}', type='{self.type.name}', palette='{self.palette.name}')"


class TextChartInfo(VisualizationInfo):
    """This represents a visualization using text, which uses only a single string."""

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name, VisualizationType.TEXT)

    def __repr__(self):
        """Returns the string representation of the object."""
        return f"TextChartInfo(name='{self.name}, 'type='{self.type}')"


class LineChartInfo(VisualizationInfo):
    """This represents a visualization using a line chart."""

    x_axis_label: str
    y_axis_label: str

    def __init__(
        self,
        name: str,
        x_axis_label: str = None,
        y_axis_label: str = None,
        palette: ColorPalette = ColorPalette.DEFAULT,
    ):
        super().__init__(name, VisualizationType.LINE, palette)
        if x_axis_label is None:
            x_axis_label = ""
        if y_axis_label is None:
            y_axis_label = ""

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label

    def __repr__(self):
        """Returns the string representation of the object."""
        return (
            f"LineChartInfo(name='{self.name}, 'type='{self.type}', x_axis_label='{self.x_axis_label}', "
            f"y_axis_label='{self.y_axis_label}')"
        )


class BarChartInfo(VisualizationInfo):
    """This represents a visualization using a bar chart."""

    def __init__(
        self,
        name: str,
        palette: ColorPalette = ColorPalette.DEFAULT,
        visualization_type: VisualizationType = VisualizationType.BAR,
    ):
        if visualization_type not in (
            VisualizationType.BAR,
            VisualizationType.RADIAL_BAR,
        ):
            raise ValueError("Visualization type for BarChartInfo must be BAR or RADIAL_BAR")
        super().__init__(name, visualization_type, palette)

    def __repr__(self):
        """Returns the string representation of the object."""
        return f"BarChartInfo(name='{self.name}', type='{self.type}')"


class MatrixChartInfo(VisualizationInfo):
    """This represents a visualization using a matrix."""

    header: str
    row_header: str
    column_header: str

    # pylint: disable=too-many-arguments; Requires refactor
    def __init__(
        self,
        name: str,
        header: str = None,
        row_header: str = None,
        column_header: str = None,
        palette: ColorPalette = ColorPalette.DEFAULT,
    ):
        super().__init__(name, VisualizationType.MATRIX, palette)
        if header is not None:
            self.header = header
        if row_header is not None:
            self.row_header = row_header
        if column_header is not None:
            self.column_header = column_header

    def __repr__(self):
        """Returns the string representation of the object."""
        return (
            f"MatrixChartInfo(name='{self.name}', type='{self.type}', header='{self.header}', row_header='"
            f"{self.row_header}', column_header='{self.column_header}')"
        )


_Metric = TypeVar("_Metric", bound=MetricEntity)
_VisualizationInfo = TypeVar("_VisualizationInfo", bound=VisualizationInfo)


class MetricsGroup(Generic[_Metric, _VisualizationInfo]):
    """This class aggregates a list of metric entities and defines how this group will be visualized on the UI.

    This class is the parent class to the different types of MetricsGroup that each represent a different type of chart
    in the UI.

    Example:
        An accuracy as a metrics group
            >>> acc = ScoreMetric("Accuracy", 0.5)
            >>> visual_info = BarChartInfo("Accuracy", visualization_type=_VisualizationInfo.BAR)  # show as radial bar
            >>> metrics_group = BarMetricsGroup([acc], visual_info)

        Loss curves as a metrics group
            >>> train_loss = CurveMetric("Train loss", xs=[0, 1, 2], ys=[5, 3, 1])
            >>> val_loss = CurveMetric("Validation", xs=[0, 1, 2], ys=[6, 4, 2])
            >>> visual_info = LineChartInfo("Loss curve", x_axis_label="# epoch", y_axis_label="Loss")
            >>> metrics_group = LineMetricsGroup([train_loss, val_loss], visual_info)
    """

    def __init__(self, metrics: Sequence[_Metric], visualization_info: _VisualizationInfo):
        if metrics is None or len(metrics) == 0:
            raise ValueError("Metrics cannot be None or empty")
        if visualization_info is None:
            raise ValueError("visualization_info cannot be None")
        self.metrics = metrics
        self.visualization_info = visualization_info


class MatrixMetricsGroup(MetricsGroup[MatrixMetric, MatrixChartInfo]):
    """This class represent a matrix chart in the UI.

    Multiple matrices can be displayed in the same chart.
    """

    def __init__(self, metrics: Sequence[MatrixMetric], visualization_info: MatrixChartInfo):
        super().__init__(metrics=metrics, visualization_info=visualization_info)


class LineMetricsGroup(MetricsGroup[CurveMetric, LineChartInfo]):
    """This class represent a line chart in the UI.

    Multiple lines can be displayed in a single chart.
    """

    def __init__(self, metrics: Sequence[CurveMetric], visualization_info: LineChartInfo):
        super().__init__(metrics=metrics, visualization_info=visualization_info)


class BarMetricsGroup(MetricsGroup[Union[ScoreMetric, CountMetric], BarChartInfo]):
    """This class represent a bar or radial bar chart in the UI.

    Each metric in the metrics group represents the value of a single bar/radial bar in the chart.
    """

    def __init__(
        self,
        metrics: Sequence[Union[ScoreMetric, CountMetric]],
        visualization_info: BarChartInfo,
    ):
        super().__init__(metrics=metrics, visualization_info=visualization_info)


class TextMetricsGroup(
    MetricsGroup[
        Union[ScoreMetric, CountMetric, InfoMetric, DateMetric, DurationMetric],
        TextChartInfo,
    ]
):
    """This class represent a text chart in the UI.

    Text charts contain only one metric, which can be of type CountMetric, ScoreMetric, DateMetric, DurationMetric or
    InfoMetric.
    """

    def __init__(
        self,
        metrics: Sequence[Union[ScoreMetric, CountMetric, InfoMetric, DateMetric, DurationMetric]],
        visualization_info: TextChartInfo,
    ):
        if not len(metrics) == 1:
            raise ValueError(
                "A text metrics group can contain only a single "
                "ScoreMetric, CountMetric, InfoMetric, DateMetric or "
                "DurationMetric."
            )
        super().__init__(metrics=metrics, visualization_info=visualization_info)


class Performance:
    """This performance class wraps the statistics of an entity (e.g., Model, Resultset).

    Args:
        score: the performance score. This will be the point of
            comparison between two performances.
        dashboard_metrics: (optional) additional statistics, containing
            charts, curves, and other additional info.
    """

    def __init__(self, score: ScoreMetric, dashboard_metrics: Optional[List[MetricsGroup]] = None):
        if not isinstance(score, ScoreMetric):
            raise ValueError(f"Expected score to be of type `ScoreMetric`, got type `{type(score)}` instead.")
        self._score: ScoreMetric = score
        self.dashboard_metrics: List[MetricsGroup] = [] if dashboard_metrics is None else dashboard_metrics

    @property
    def score(self):
        """Return the score metric."""
        return self._score

    def __eq__(self, other: object) -> bool:
        """Returns True if self and other have the same score and dashboard metrics."""
        if not isinstance(other, Performance):
            return False
        return self.score == other.score

    def __repr__(self):
        """Returns a string representation of the performance."""
        return f"Performance(score: {self.score.value}, dashboard: ({len(self.dashboard_metrics)} metric groups))"


class NullPerformance(Performance):
    """This is used to represent 'Performance not found'."""

    def __init__(self) -> None:
        super().__init__(score=ScoreMetric(name="Null score", value=0.0))

    def __repr__(self):
        """Returns a string representation of the performance."""
        return "NullPerformance()"

    def __eq__(self, other):
        """Returns True if other is a NullPerformance."""
        return isinstance(other, NullPerformance)


class MultiScorePerformance(Performance):
    """This class can be used in tasks where performance is measured by multiple metrics.

    Args:
        primary_score: The main performance score.
        additional_metrics: List of additional scores. When no primary
            score is provided, the first additional score takes priority
            as the main project score.
        dashboard_metrics: (optional) additional statistics, containing
            charts, curves, and other additional info.
    """

    def __init__(
        self,
        primary_score: Optional[ScoreMetric] = None,
        additional_scores: Optional[List[ScoreMetric]] = None,
        dashboard_metrics: Optional[List[MetricsGroup]] = None,
    ):
        assert primary_score is not None or (
            additional_scores is not None and len(additional_scores) > 0
        ), "Provide at least one primary or additional score."

        self._primary_score = primary_score
        self._additional_scores: List[ScoreMetric] = [] if additional_scores is None else additional_scores
        self.dashboard_metrics: List[MetricsGroup] = [] if dashboard_metrics is None else dashboard_metrics

        if self.primary_score is None:
            super().__init__(self.additional_scores[0], dashboard_metrics)
        else:
            super().__init__(self.primary_score, dashboard_metrics)

    @property
    def primary_score(self) -> Optional[ScoreMetric]:
        """Return the primary score metric."""
        return self._primary_score

    @property
    def additional_scores(self) -> List[ScoreMetric]:
        """Return the additional score metrics."""
        return self._additional_scores

    def __eq__(self, other: object) -> bool:
        """Returns True if the other object is a MultiScorePerformance with the same primary and additional scores."""
        if not isinstance(other, MultiScorePerformance):
            return False
        return self.primary_score == other.primary_score and self.additional_scores == other.additional_scores

    def __repr__(self):
        """Returns the representation of the performance."""
        return (
            f"MultiScorePerformance(score: {self.score.value}, primary_metric: {self.primary_score}, "
            f"additional_metrics: ({len(self.additional_scores)} metrics), "
            f"dashboard: ({len(self.dashboard_metrics)} metric groups))"
        )
