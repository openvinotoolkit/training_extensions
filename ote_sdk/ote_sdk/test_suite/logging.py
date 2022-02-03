# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging


def get_logger():
    logger_name = ".".join(__name__.split(".")[:-1])
    return logging.getLogger(logger_name)


def set_log_level(level, recursive=False):
    logger = get_logger()
    logger.setLevel(level)
    if not recursive:
        return
    while logger:
        logger.setLevel(level)
        logger = logger.parent
