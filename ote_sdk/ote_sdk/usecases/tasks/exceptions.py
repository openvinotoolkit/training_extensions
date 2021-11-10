"""This module contains the exceptions for the tasks. """


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


class UnrecoverableTaskException(Exception):
    """Exception for when task is in an unrecoverable state."""

    def __init__(self, message="Unrecoverable task exception"):
        # pylint: disable=W0235
        super().__init__(message)


class OOMException(UnrecoverableTaskException):
    """Exception for when task is out of memory."""

    def __init__(self, message="Out of memory exception"):
        super().__init__(message)


class TrainingStallException(UnrecoverableTaskException):
    """Exception for when training should be stalled."""

    def __init__(self, message="Training stalling exception"):
        super().__init__(message)
