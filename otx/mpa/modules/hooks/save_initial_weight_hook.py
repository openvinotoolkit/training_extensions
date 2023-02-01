# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SaveInitialWeightHook(Hook):
    def __init__(self, save_path, **kwargs):
        self.save_path = save_path
        self.args = kwargs

    def before_run(self, runner):
        runner.logger.info("Saving weight before training")
        runner.save_checkpoint(
            self.save_path, filename_tmpl="weights.pth", save_optimizer=False, create_symlink=False, **self.args
        )
