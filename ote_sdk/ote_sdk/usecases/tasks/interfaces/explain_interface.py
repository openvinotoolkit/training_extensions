"""This module contains the interface class for tasks. """


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters


class IExplainTask(metaclass=abc.ABCMeta):
    """
    A base interface for explain task.
    """

    @abc.abstractmethod
    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: InferenceParameters,
    ) -> DatasetEntity:
        """
        This is the method that is called upon explanation.

        :param dataset: The input dataset to perform the explain on.
        :param explain_parameters: The parameters to use for the explain.
        :return: The results of the explain, such as saliency_map or feature_map
        """
        raise NotImplementedError