"""This module contains the exceptions for the tasks."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


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
