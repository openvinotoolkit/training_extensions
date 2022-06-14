"""Score reporting callback"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Optional

from ote_sdk.entities.train_parameters import TrainParameters
from pytorch_lightning import Callback


class ScoreReportingCallback(Callback):
    """
    Callback for reporting score.
    """

    def __init__(self, parameters: Optional[TrainParameters] = None) -> None:
        if parameters is not None:
            self.score_reporting_callback = parameters.update_progress
        else:
            self.score_reporting_callback = None

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        If score exists in trainer.logged_metrics, report the score.
        """
        if self.score_reporting_callback is not None:
            score = None
            metric = getattr(self.score_reporting_callback, 'metric', None)
            if metric in trainer.logged_metrics:
                score = float(trainer.logged_metrics[metric])
            self.score_reporting_callback(progress=0, score=score)
