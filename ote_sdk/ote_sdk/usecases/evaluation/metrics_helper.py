""" This module contains the helper functions which can be called directly by
algorithm implementers to obtain the metrices """

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.usecases.evaluation.accuracy import Accuracy
from ote_sdk.usecases.evaluation.averaging import MetricAverageMethod
from ote_sdk.usecases.evaluation.dice import DiceAverage
from ote_sdk.usecases.evaluation.f_measure import FMeasure


class MetricsHelper:
    """
    Contains metrics computation functions.
    TODO: subject for refactoring.
    """

    @staticmethod
    def compute_f_measure(
        resultset: ResultSetEntity,
        vary_confidence_threshold: bool = False,
        vary_nms_threshold: bool = False,
        cross_class_nms: bool = False,
    ) -> FMeasure:
        """
        Compute the F-Measure on a resultset given some parameters.

        :param resultset: The resultset used to compute f-measure
        :param vary_confidence_threshold: Flag specifying whether f-measure shall be computed for different confidence
                                          threshold values
        :param vary_nms_threshold: Flag specifying whether f-measure shall be computed for different NMS
                                   threshold values
        :param cross_class_nms: Whether non-max suppression should be applied cross-class
        :return: FMeasure object
        """
        return FMeasure(
            resultset, vary_confidence_threshold, vary_nms_threshold, cross_class_nms
        )

    @staticmethod
    def compute_dice_averaged_over_pixels(
        resultset: ResultSetEntity,
        average: MetricAverageMethod = MetricAverageMethod.MACRO,
    ) -> DiceAverage:
        """
        Compute the Dice average on a resultset, averaged over the pixels.

        :param resultset: The resultset used to compute the Dice average
        :param average: The averaging method, either MICRO or MACRO
        :return: DiceAverage object
        """
        return DiceAverage(resultset=resultset, average=average)

    @staticmethod
    def compute_accuracy(
        resultset: ResultSetEntity,
        average: MetricAverageMethod = MetricAverageMethod.MICRO,
    ) -> Accuracy:
        """
        Compute the Accuracy on a resultset, averaged over the different label groups.

        :param resultset: The resultset used to compute the accuracy
        :param average: The averaging method, either MICRO or MACRO
        :return: Accuracy object
        """
        return Accuracy(resultset=resultset, average=average)
