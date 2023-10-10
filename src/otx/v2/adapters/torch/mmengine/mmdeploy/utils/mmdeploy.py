"""Functions for mmdeploy adapters."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import platform


def is_mmdeploy_enabled() -> bool:
    """Checks if the 'mmdeploy' Python module is installed and available for use.

    Returns:
    -------
        bool: True if 'mmdeploy' is installed, False otherwise.

    Example:
    -------
        >>> is_mmdeploy_enabled()
        True
    """
    if platform.system() not in ("Linux", "Windows"):
        return False
    return importlib.util.find_spec("mmdeploy") is not None


if is_mmdeploy_enabled():
    # fmt: off
    # FIXME: openvino pot library adds stream handlers to root logger
    # which makes annoying duplicated logging
    from mmdeploy.utils import get_root_logger
    get_root_logger().propagate = False
    # fmt: on
