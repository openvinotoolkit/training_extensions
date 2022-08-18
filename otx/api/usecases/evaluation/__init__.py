"""
Evaluation metrics

.. automodule:: otx.api.usecases.evaluation.accuracy
   :members:
   :undoc-members:

.. automodule:: otx.api.usecases.evaluation.basic_operations
   :members:
   :undoc-members:

.. automodule:: otx.api.usecases.evaluation.dice
   :members:
   :undoc-members:

.. automodule:: otx.api.usecases.evaluation.f_measure
   :members:
   :undoc-members:

.. automodule:: otx.api.usecases.evaluation.get_performance_interface
   :members:
   :undoc-members:

.. automodule:: otx.api.usecases.evaluation.metrics_helper
   :members:
   :undoc-members:

"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .accuracy import Accuracy
from .averaging import MetricAverageMethod
from .basic_operations import (
    get_intersections_and_cardinalities,
    intersection_box,
    intersection_over_union,
    precision_per_class,
    recall_per_class,
)
from .dice import DiceAverage
from .f_measure import FMeasure
from .metrics_helper import MetricsHelper

__all__ = [
    "Accuracy",
    "MetricAverageMethod",
    "DiceAverage",
    "FMeasure",
    "intersection_box",
    "intersection_over_union",
    "MetricsHelper",
    "precision_per_class",
    "recall_per_class",
    "get_intersections_and_cardinalities",
]
