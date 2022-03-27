# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from pytorch_lightning.callbacks import Callback


class StopCallback(Callback):
    """Stop training callback"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def on_batch_end(self, trainer, pl_module):
        trainer.should_stop = self.should_stop

    def stop(self):
        self.should_stop = True

    def reset(self):
        self.should_stop = False

    def check_stop(self):
        return self.should_stop
