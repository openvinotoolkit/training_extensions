"""This module contains the interface class for tasks."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Dict

import numpy as np

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters


class IInferenceTask(metaclass=abc.ABCMeta):
    """A base interface class for a task."""

    @abc.abstractmethod
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: InferenceParameters,
    ) -> DatasetEntity:
        """This is the method that is called upon inference.

        This happens when the user wants to analyse a sample
        or multiple samples need to be analysed.

        Args:
            dataset: The input dataset to perform the analysis on.
            inference_parameters: The parameters to use for the
                analysis.

        Returns:
            The results of the analysis.
        """
        raise NotImplementedError


class IRawInference(metaclass=abc.ABCMeta):
    """A base interface class for raw inference tasks."""

    @abc.abstractmethod
    def raw_infer(
        self,
        input_tensors: Dict[str, np.ndarray],
        output_tensors: Dict[str, np.ndarray],
    ):
        """This is the method that is called to run a neural network over a set of tensors.

        This method takes as input/output the tensors which are directly fed to the neural network,
        and does not include any additional pre- and post-processing of the inputs and outputs.

        Args:
            input_tensors: Dictionary containing the input tensors.
            output_tensors: Dictionary to be filled by the task with the
                output tensors.
        """
        raise NotImplementedError
