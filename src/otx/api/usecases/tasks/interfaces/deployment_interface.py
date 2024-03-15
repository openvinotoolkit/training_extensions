"""This module contains the interface class for tasks that can deploy their models."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc

from otx.api.entities.model import ModelEntity


class IDeploymentTask(metaclass=abc.ABCMeta):
    """A base interface class for tasks which can deploy their models."""

    @abc.abstractmethod
    def deploy(self, output_model: ModelEntity) -> None:
        """This method defines the interface for deploy.

        Args:
            output_model: Output model
        """
        raise NotImplementedError
