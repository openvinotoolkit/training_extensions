"""Module for defining custom logger."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ruff: noqa: PLW0603

import functools
import logging
import os
import sys
from typing import Callable

import torch.distributed as dist

# __all__ = ['config_logger', 'get_log_dir', 'get_logger']
__all__ = ["config_logger", "get_log_dir"]

_LOGGING_FORMAT = "%(asctime)s | %(levelname)s : %(message)s"
_LOG_DIR = None
_FILE_HANDLER = None
_CUSTOM_LOG_LEVEL = 31

LEVEL = logging.INFO

logging.addLevelName(_CUSTOM_LOG_LEVEL, "LOG")


def _get_logger():
    logger = logging.getLogger("otx")
    logger.propagate = False

    def logger_print(message, *args, **kws):
        if logger.isEnabledFor(_CUSTOM_LOG_LEVEL):
            logger.log(_CUSTOM_LOG_LEVEL, message, *args, **kws)

    logger.print = logger_print

    logger.setLevel(LEVEL)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOGGING_FORMAT))

    logger.addHandler(console)

    return logger


_logger = _get_logger()

# # to expose supported APIs
# _override_methods = ['setLevel', 'addHandler', 'addFilter', 'info',
#                      'warning', 'error', 'critical', 'print']
# for fn in _override_methods:
#     locals()[fn] = getattr(_logger, fn)
#     __all__.append(fn)


def config_logger(log_file, level="WARNING"):
    """A function that configures the logging system.

    :param log_file: str, a string representing the path to the log file.
    :param level: str, a string representing the log level. Default is "WARNING".
    :return: None
    """
    global _LOG_DIR, _FILE_HANDLER  # pylint: disable=global-statement
    if _FILE_HANDLER is not None:
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    _LOG_DIR = os.path.dirname(log_file)
    os.makedirs(_LOG_DIR, exist_ok=True)
    file = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file.setFormatter(logging.Formatter(_LOGGING_FORMAT))
    _FILE_HANDLER = file
    _logger.addHandler(file)
    _logger.setLevel(_get_log_level(level))


def _get_log_level(level):
    # sanity checks
    if level is None:
        return None

    # get level number
    level_number = logging.getLevelName(level.upper())
    if level_number not in [0, 10, 20, 30, 40, 50, _CUSTOM_LOG_LEVEL]:
        msg = f"Log level must be one of DEBUG/INFO/WARN/ERROR/CRITICAL/LOG, but {level} is given."
        raise ValueError(msg)

    return level_number


def get_log_dir():
    """A function that retrieves the directory path of the log file.

    :return: str, a string representing the directory path of the log file.
    """
    return _LOG_DIR


class _DummyLogger(logging.Logger):
    def debug(self, message, *args, **kws):
        pass

    def info(self, message, *args, **kws):
        pass

    def warning(self, message, *args, **kws):
        pass

    def critical(self, message, *args, **kws):
        pass

    def error(self, message, *args, **kws):
        pass


def local_master_only(func: Callable) -> Callable:
    """A decorator that allows a function to be executed only by the local master process in distributed training setup.

    Args:
        func: the function to be decorated.

    Returns:
        A wrapped function that can only be executed by the local master process.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
        local_rank = 0
        if dist.is_available() and dist.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            return func(*args, **kwargs)

    return wrapper


# apply decorator @local_master_only to the lower severity logging functions
_logging_methods = ["print", "debug", "info", "warning"]
for fn in _logging_methods:
    setattr(_logger, fn, local_master_only(getattr(_logger, fn)))


def get_logger():
    """Return logger."""
    # if dist.is_available() and dist.is_initialized():
    #     rank = dist.get_rank()
    # else:
    #     rank = 0
    # if rank == 0:
    #     return _logger
    # return _DummyLogger('dummy')
    return _logger
