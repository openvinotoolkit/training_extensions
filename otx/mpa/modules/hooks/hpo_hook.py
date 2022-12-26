# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class HPOHook(Hook):

    def __init__(self, hp_config, metric):
        self.hp_config = hp_config
        self.metric = metric

    def after_train_epoch(self, runner):
        import hpopt

        if self.metric in runner.log_buffer.output:
            score = runner.log_buffer.output[self.metric]
        else:
            score = -999999

        if hpopt.report(config=self.hp_config, score=score) == hpopt.Status.STOP:
            runner._max_epochs = 1
