"""Helper functions for computing metrics.

This module contains the helper functions which can be called directly by algorithm implementers to obtain the metrics.
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.evaluation.accuracy import Accuracy
from otx.api.usecases.evaluation.anomaly_metrics import (
    AnomalyDetectionScores,
    AnomalySegmentationScores,
)
from otx.api.usecases.evaluation.averaging import MetricAverageMethod
from otx.api.usecases.evaluation.dice import DiceAverage
from otx.api.usecases.evaluation.f_measure import FMeasure


class MetricsHelper:
    """Contains metrics computation functions.

    TODO: subject for refactoring.
    """

    @staticmethod
    def compute_f_measure(
        resultset: ResultSetEntity,
        vary_confidence_threshold: bool = False,
        vary_nms_threshold: bool = False,
        cross_class_nms: bool = False,
    ) -> FMeasure:
        """Compute the F-Measure on a resultset given some parameters.

        Args:
            resultset: The resultset used to compute f-measure
            vary_confidence_threshold: Flag specifying whether f-measure
                shall be computed for different confidence threshold
                values
            vary_nms_threshold: Flag specifying whether f-measure shall
                be computed for different NMS threshold values
            cross_class_nms: Whether non-max suppression should be
                applied cross-class

        Returns:
            FMeasure object
        """
        return FMeasure(resultset, vary_confidence_threshold, vary_nms_threshold, cross_class_nms)

    @staticmethod
    def compute_dice_averaged_over_pixels(
        resultset: ResultSetEntity,
        average: MetricAverageMethod = MetricAverageMethod.MACRO,
    ) -> DiceAverage:
        """Compute the Dice average on a resultset, averaged over the pixels.

        Args:
            resultset: The resultset used to compute the Dice average
            average: The averaging method, either MICRO or MACRO

        Returns:
            DiceAverage object
        """
        return DiceAverage(resultset=resultset, average=average)

    @staticmethod
    def compute_accuracy(
        resultset: ResultSetEntity,
        average: MetricAverageMethod = MetricAverageMethod.MICRO,
    ) -> Accuracy:
        """Compute the Accuracy on a resultset, averaged over the different label groups.

        Args:
            resultset: The resultset used to compute the accuracy
            average: The averaging method, either MICRO or MACRO

        Returns:
            Accuracy object
        """
        return Accuracy(resultset=resultset, average=average)

    @staticmethod
    def compute_anomaly_segmentation_scores(
        resultset: ResultSetEntity,
    ) -> AnomalySegmentationScores:
        """Compute the anomaly localization performance metrics on an anomaly segmentation resultset.

        Args:
            resultset: The resultset used to compute the metrics

        Returns:
            AnomalyLocalizationScores object
        """
        return AnomalySegmentationScores(resultset)

    @staticmethod
    def compute_anomaly_detection_scores(
        resultset: ResultSetEntity,
    ) -> AnomalyDetectionScores:
        """Compute the anomaly localization performance metrics on an anomaly detection resultset.

        Args:
            resultset: The resultset used to compute the metrics

        Returns:
            AnomalyLocalizationScores object
        """
        return AnomalyDetectionScores(resultset)
