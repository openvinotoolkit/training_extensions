""" This module contains the implementations of performance providers for multi-score anomaly metrics. """

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional

from ote_sdk.entities.metrics import (
    MetricsGroup,
    MultiScorePerformance,
    Performance,
    ScoreMetric,
)
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.usecases.evaluation.averaging import MetricAverageMethod
from ote_sdk.usecases.evaluation.dice import DiceAverage
from ote_sdk.usecases.evaluation.f_measure import FMeasure
from ote_sdk.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from ote_sdk.utils.dataset_utils import (
    contains_anomalous_images,
    split_local_global_resultset,
)


class AnomalySegmentationScores(IPerformanceProvider):
    """
    This class provides the MultiScorePerformance object for anomaly segmentation resultsets.
    The returned performance object contains the local (pixel-level) performance metric as the main score if local
    annotations are available. The global (image-level) performance metric is included as additional metric.

    :param resultset: ResultSet that scores will be computed for
    """

    def __init__(self, resultset: ResultSetEntity):
        self.local_score: Optional[ScoreMetric] = None
        self.dashboard_metrics: Optional[List[MetricsGroup]] = None

        global_resultset, local_resultset = split_local_global_resultset(resultset)

        global_metric = FMeasure(resultset=global_resultset)
        global_performance = global_metric.get_performance()
        self.global_score = global_performance.score

        if contains_anomalous_images(local_resultset.ground_truth_dataset):
            local_metric = DiceAverage(
                resultset=local_resultset, average=MetricAverageMethod.MICRO
            )
            local_performance = local_metric.get_performance()
            self.local_score = local_performance.score
            self.dashboard_metrics = local_performance.dashboard_metrics

    def get_performance(self) -> Performance:
        return MultiScorePerformance(
            self.local_score, [self.global_score], self.dashboard_metrics
        )
