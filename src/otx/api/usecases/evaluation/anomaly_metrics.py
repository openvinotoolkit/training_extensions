"""This module contains the implementations of performance providers for multi-score anomaly metrics."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC
from typing import List, Optional

from otx.api.entities.metrics import (
    MetricsGroup,
    MultiScorePerformance,
    Performance,
    ScoreMetric,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.evaluation.averaging import MetricAverageMethod
from otx.api.usecases.evaluation.dice import DiceAverage
from otx.api.usecases.evaluation.f_measure import FMeasure
from otx.api.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from otx.api.utils.dataset_utils import (
    contains_anomalous_images,
    split_local_global_resultset,
)


class AnomalyLocalizationPerformance(MultiScorePerformance):
    """Anomaly specific MultiScorePerformance.

    This class implements a special case of the MultiScorePerformance, specific for anomaly tasks that perform
    anomaly localization (detection/segmentation), in addition to anomaly classification.

    Args:
        global_score: Image-level performance metric.
        local_score: Pixel- or bbox-level performance metric, depending
            on the task type.
        dashboard_metrics: (optional) additional statistics, containing
            charts, curves, and other additional info.
    """

    def __init__(
        self,
        global_score: ScoreMetric,
        local_score: Optional[ScoreMetric],
        dashboard_metrics: Optional[List[MetricsGroup]],
    ):
        super().__init__(
            primary_score=local_score,
            additional_scores=[global_score],
            dashboard_metrics=dashboard_metrics,
        )
        self._global_score = global_score
        self._local_score = local_score

    @property
    def global_score(self):
        """Return the global (image-level) score metric."""
        return self._global_score

    @property
    def local_score(self):
        """Return the local (pixel-/bbox-level) score metric."""
        return self._local_score


class AnomalyLocalizationScores(IPerformanceProvider, ABC):
    """AnomalyLocalizationPerformance object for anomaly segmentation and anomaly detection tasks.

    Depending on the subclass, the `get_performance` method returns an AnomalyLocalizationPerformance object with the
    pixel- or bbox-level metric as the primary score. The global (image-level) performance metric is included as an
    additional metric.

    Args:
        resultset: ResultSet that scores will be computed for
    """

    def __init__(self, resultset: ResultSetEntity):
        self.local_score: Optional[ScoreMetric] = None
        self.dashboard_metrics: List[MetricsGroup] = []

        global_resultset, local_resultset = split_local_global_resultset(resultset)

        global_metric = FMeasure(resultset=global_resultset)
        global_performance = global_metric.get_performance()
        self.global_score = global_performance.score
        self.dashboard_metrics += global_performance.dashboard_metrics

        if contains_anomalous_images(local_resultset.ground_truth_dataset):
            local_metric = self._get_local_metric(local_resultset)
            local_performance = local_metric.get_performance()
            self.local_score = local_performance.score
            self.dashboard_metrics += local_performance.dashboard_metrics

    @staticmethod
    def _get_local_metric(local_resultset: ResultSetEntity) -> IPerformanceProvider:
        """Return the local performance metric for the resultset."""
        raise NotImplementedError

    def get_performance(self) -> Performance:
        """Return the performance object for the resultset."""
        return AnomalyLocalizationPerformance(
            global_score=self.global_score,
            local_score=self.local_score,
            dashboard_metrics=self.dashboard_metrics,
        )


class AnomalySegmentationScores(AnomalyLocalizationScores):
    """Performance provider for anomaly segmentation tasks."""

    @staticmethod
    def _get_local_metric(local_resultset: ResultSetEntity) -> IPerformanceProvider:
        return DiceAverage(resultset=local_resultset, average=MetricAverageMethod.MICRO)


class AnomalyDetectionScores(AnomalyLocalizationScores):
    """Performance provider for anomaly detection tasks."""

    @staticmethod
    def _get_local_metric(local_resultset: ResultSetEntity) -> IPerformanceProvider:
        return FMeasure(resultset=local_resultset)
