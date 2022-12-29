# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional, Callable

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SaveInitialWeightHook(Hook):
    def __init__(self, save_path, file_name: str = "weights.pth", after_save_func: Optional[Callable] = None, **kwargs):
        self._save_path = save_path
        self._file_name = file_name
        self._after_save_func = after_save_func
        self._args = kwargs

    def before_run(self, runner):
        runner.logger.info('Saving weight before training')
        runner.save_checkpoint(
            self._save_path,
            filename_tmpl=self._file_name,
            save_optimizer=False,
            create_symlink=False,
            **self._args
        )

        if self._after_save_func is not None:
            self._after_save_func()
