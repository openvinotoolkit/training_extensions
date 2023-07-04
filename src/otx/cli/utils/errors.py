"""Utils for CLI errors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


class CliException(Exception):
    """Custom exception class for CLI."""


class ConfigValueError(CliException):
    """Configuration value is not suitable for CLI."""

    def __init__(self, message):
        super().__init__(message)


class NotSupportedError(CliException):
    """Not supported error."""

    def __init__(self, message):
        super().__init__(message)


class FileNotExistError(CliException):
    """Not exist given configuration."""

    def __init__(self, message):
        super().__init__(message)
