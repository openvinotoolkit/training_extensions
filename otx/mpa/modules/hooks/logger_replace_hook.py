# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class LoggerReplaceHook(Hook):
    """replace logger in the runner to the MPA logger.
    DO NOT INCLUDE this hook to the recipe directly.
    mpa will add this hook to all recipe internally.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_run(self, runner):
        runner.logger = logger
        logger.info("logger in the runner is replaced to the MPA logger")
