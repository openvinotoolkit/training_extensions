""" This module contains interface for performance providers. """

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

from ote_sdk.entities.metrics import Performance


class IPerformanceProvider(metaclass=abc.ABCMeta):
    """
    Interface for performance provider.
    TODO: subject for refactoring.
    """

    @abc.abstractmethod
    def get_performance(self) -> Performance:
        """
        Returns the computed performance
        """
        raise NotImplementedError
