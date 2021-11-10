"""This module contains the interface class for tasks. """


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
from typing import Dict

import numpy as np

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters


class IInferenceTask(metaclass=abc.ABCMeta):
    """
    A base interface class for a task.
    """

    @abc.abstractmethod
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: InferenceParameters,
    ) -> DatasetEntity:
        """
        This is the method that is called upon inference.
        This happens when the user wants to analyse a sample
        or multiple samples need to be analysed.

        :param dataset: The input dataset to perform the analysis on.
        :param inference_parameters: The parameters to use for the analysis.
        :return: The results of the analysis.
        """
        raise NotImplementedError


class IRawInference(metaclass=abc.ABCMeta):
    """
    A base interface class for raw inference tasks.
    """

    @abc.abstractmethod
    def raw_infer(
        self,
        input_tensors: Dict[str, np.ndarray],
        output_tensors: Dict[str, np.ndarray],
    ):
        """
        This is the method that is called to run a neural network over a set of tensors.
        This method takes as input/output the tensors which are directly fed to the neural network,
        and does not include any additional pre- and post-processing of the inputs and outputs.

        :param input_tensors: Dictionary containing the input tensors.
        :param output_tensors: Dictionary to be filled by the task with the output tensors.
        """
        raise NotImplementedError
