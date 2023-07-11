"""This module contains the interface class for tasks that can compute performance."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Optional

from otx.api.entities.resultset import ResultSetEntity


class IEvaluationTask(metaclass=abc.ABCMeta):
    """A base interface class for tasks which can compute performance on a resultset."""

    @abc.abstractmethod
    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Compute performance metrics for a given set of results.

        The task may use at its discretion the most appropriate metrics for the evaluation (for instance,
        average precision for classification, DICE for segmentation, etc).
        The performance will be stored directly to output_resultset.performance

        Args:
            output_resultset (ResultSetEntity): The set of results which must be
                evaluated.
            evaluation_metric (Optional[str]): the evaluation metric used to compute the
                performance
        """
        raise NotImplementedError
