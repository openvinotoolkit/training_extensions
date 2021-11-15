"""This module contains the interface class for tasks that can compute performance. """


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

import abc
from typing import Optional

from ote_sdk.entities.resultset import ResultSetEntity


class IEvaluationTask(metaclass=abc.ABCMeta):
    """
    A base interface class for tasks which can compute performance on a resultset.
    """

    @abc.abstractmethod
    def evaluate(
        self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None
    ):
        """
        Compute performance metrics for a given set of results.
        The task may use at its discretion the most appropriate metrics for the evaluation (for instance,
        average precision for classification, DICE for segmentation, etc).
        The performance will be stored directly to output_resultset.performance

        :param output_resultset: The set of results which must be evaluated.
        :param evaluation_metric: the evaluation metric used to compute the performance
        """
        raise NotImplementedError
