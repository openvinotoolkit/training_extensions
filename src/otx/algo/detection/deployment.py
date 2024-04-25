"""Functions for mmdeploy adapters."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib


def is_mmdeploy_enabled() -> bool:
    """Checks if the 'mmdeploy' Python module is installed and available for use.

    Returns:
        bool: True if 'mmdeploy' is installed, False otherwise.

    Example:
        >>> is_mmdeploy_enabled()
        True
    """
    return importlib.util.find_spec("mmdeploy") is not None
