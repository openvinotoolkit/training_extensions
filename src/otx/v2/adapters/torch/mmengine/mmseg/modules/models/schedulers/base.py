"""Base scalar scheduler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABCMeta, abstractmethod


class BaseScalarScheduler(metaclass=ABCMeta):
    """Base scalar scheduler."""

    def __call__(self, step, epoch_size) -> float:
        """Callback function of BaseScalarScheduler."""
        return self._get_value(step, epoch_size)

    @abstractmethod
    def _get_value(self, step, epoch_size) -> float:
        raise NotImplementedError("Subclass must implement this method")
