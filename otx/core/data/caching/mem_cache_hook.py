"""Memory cache hook for logging and freezing MemCacheHandler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner.hooks import HOOKS, Hook

from .mem_cache_handler import MemCacheHandlerSingleton


@HOOKS.register_module()
class MemCacheHook(Hook):
    """Memory cache hook for logging and freezing MemCacheHandler."""

    def __init__(self) -> None:
        self.handler = MemCacheHandlerSingleton.get()

    def before_run(self, runner):
        """Before run, freeze the handler."""
        self.handler.freeze()

    def before_epoch(self, runner):
        """Before training, unfreeze the handler."""
        self.handler.unfreeze()

    def after_epoch(self, runner):
        """After epoch. Log the handler statistics."""
        self.handler.freeze()
        runner.logger.info(f"{self.handler}")
