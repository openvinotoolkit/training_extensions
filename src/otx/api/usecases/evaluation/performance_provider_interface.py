"""This module contains interface for performance providers."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import abc

from otx.api.entities.metrics import Performance


class IPerformanceProvider(metaclass=abc.ABCMeta):
    """Interface for performance provider.

    TODO: subject for refactoring.
    """

    @abc.abstractmethod
    def get_performance(self) -> Performance:
        """Returns the computed performance."""
        raise NotImplementedError
