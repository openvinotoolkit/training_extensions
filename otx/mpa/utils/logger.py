# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import sys

# import torch.distributed as dist

from mmcv.runner.dist_utils import master_only

# __all__ = ['config_logger', 'get_log_dir', 'get_logger']
__all__ = ['config_logger', 'get_log_dir']

_LOGGING_FORMAT = '%(asctime)s | %(levelname)s : %(message)s'
_LOG_DIR = None
_FILE_HANDLER = None
_CUSTOM_LOG_LEVEL = 31

logging.addLevelName(_CUSTOM_LOG_LEVEL, 'LOG')


def _get_logger():
    logger = logging.getLogger('mpa')
    logger.propagate = False

    def print(message, *args, **kws):
        if logger.isEnabledFor(_CUSTOM_LOG_LEVEL):
            logger.log(_CUSTOM_LOG_LEVEL, message, *args, **kws)

    logger.print = print

    logger.setLevel(logging.INFO)
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


def config_logger(log_file, level='WARNING'):
    global _LOG_DIR, _FILE_HANDLER
    if _FILE_HANDLER is not None:
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    _LOG_DIR = os.path.dirname(log_file)
    os.makedirs(_LOG_DIR, exist_ok=True)
    file = logging.FileHandler(log_file, mode='w', encoding='utf-8')
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
        msg = ('Log level must be one of DEBUG/INFO/WARN/ERROR/CRITICAL/LOG'
               ', but {} is given.'.format(level))
        raise ValueError(msg)

    return level_number


def get_log_dir():
    return _LOG_DIR


class _DummyLogger(logging.Logger):
    def debug(message, *args, **kws):
        pass

    def info(message, *args, **kws):
        pass

    def warning(message, *args, **kws):
        pass

    def critical(message, *args, **kws):
        pass

    def error(message, *args, **kws):
        pass


# apply decorator @master_only to the lower severity logging functions
# TODO: need to check whether it works as expected
_logging_methods = ['print', 'debug', 'info', 'warning']
for fn in _logging_methods:
    master_only(getattr(_logger, fn))


def get_logger():
    # if dist.is_available() and dist.is_initialized():
    #     rank = dist.get_rank()
    # else:
    #     rank = 0
    # if rank == 0:
    #     return _logger
    # return _DummyLogger('dummy')
    return _logger
