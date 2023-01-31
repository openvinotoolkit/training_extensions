# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
The module with common logger for all OTX training tests.
"""

import logging


def get_logger():
    """
    The function returns the common logger for all OTX
    training tests.
    """
    logger_name = ".".join(__name__.split(".")[:-1])
    return logging.getLogger(logger_name)


def set_log_level(level, recursive=False):
    """
    The function sets log level for the common logger for all
    OTX training tests.
    The parameter `level` may be either int code or string
    (e.g. 'DEBUG')
    """

    logger = get_logger()
    logger.setLevel(level)
    if not recursive:
        return
    while logger:
        logger.setLevel(level)
        logger = logger.parent
