"""This module contains the interface class for tasks."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters


class IExplainTask(metaclass=abc.ABCMeta):
    """A base interface for explain task."""

    @abc.abstractmethod
    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: ExplainParameters,
    ) -> DatasetEntity:
        """This is the method that is called upon explanation.

        Args:
            dataset: The input dataset to perform the explain on.
            explain_parameters: The parameters to use for the explain.

        Returns:
            The results of the explanation.
        """
        raise NotImplementedError
