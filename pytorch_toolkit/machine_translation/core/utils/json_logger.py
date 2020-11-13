"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import json
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase

class JSONLogger(LightningLoggerBase):
    def __init__(self, path=None):
        super().__init__()
        self.path = path
        self.log_history = []

    def name(self):
        return 'JSONLogger'

    def experiment(self):
        pass

    def version(self):
        return '0.1'

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics["step"] = step
        self.log_history.append(metrics)

    def save(self):
        super().save()
        if len(self.log_history) and self.path is not None:
            with open(self.path, "w") as f:
                json.dump(self.log_history, f, indent=4)

    @rank_zero_only
    def finalize(self, status):
        pass
