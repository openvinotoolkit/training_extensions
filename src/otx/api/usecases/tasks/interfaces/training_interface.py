"""This module contains the interface class for tasks that can perform training."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters


class ITrainingTask(metaclass=abc.ABCMeta):
    """A base interface class for tasks which can perform training."""

    @abc.abstractmethod
    def save_model(self, output_model: ModelEntity):
        """Save the model currently loaded by the task to `output_model`.

        This method is for instance used to save the pre-trained weights before training
        when the task has been initialised with pre-trained weights rather than an existing model.

        Args:
            output_model (ModelEntity): Output model where the weights should be stored
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
    ):
        """Train a new model using the model currently loaded by the task.

        If training was successful, the new model should be used for subsequent calls (e.g. `optimize` or `infer`).

        The new model weights should be saved in the object `output_model`.

        The task has two choices:

         - Set the output model weights, if the task was able to improve itself (according to own measures)
         - Set the model state as failed if it failed to improve itself (according to own measures)

        Args:
            dataset (DatasetEntity): Dataset containing the training and validation splits to use for training.
            output_model (ModelEntity): Output model where the weights should be stored
            train_parameters (TrainParameters): Training parameters
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_training(self):
        """Cancels the currently running training process.

        If training is not running, do nothing.
        """
        raise NotImplementedError
