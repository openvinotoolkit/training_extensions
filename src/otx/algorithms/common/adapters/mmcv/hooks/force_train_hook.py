"""Collections of hooks for common OTX algorithms."""

# Copyright (C) 2021-2023 Intel Corporation
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

from mmcv.runner.hooks import HOOKS, Hook

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class ForceTrainModeHook(Hook):
    """Force train mode for model.

    This is a workaround of a bug in EvalHook from MMCV.
    If a model evaluation is enabled before training by setting 'start=0' in EvalHook,
    EvalHook does not put a model in a training mode again after evaluation.

    This simple hook forces to put a model in a training mode before every train epoch
    with the lowest priority.
    """

    def before_train_epoch(self, runner):
        """Make sure to put a model in a training mode before train epoch."""
        runner.model.train()
