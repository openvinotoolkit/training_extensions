# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class LightningSpeechToTextLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.reset()

    @property
    def name(self):
        return "LightningSpeechToTextLogger"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.params = params

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self.metrics.append(metrics)

    def reset(self):
        self.metrics = []

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    def version(self):
        return "0.1"
