"""Runner with cancel for common OTX algorithms."""

# Copyright (C) 2022 Intel Corporation
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

# Is based on
# * https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/epoch_based_runner.py
# * https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py

import time
import warnings
from typing import List, Optional, Sequence

import mmcv
import torch.distributed as dist
from mmcv.runner import (
    RUNNERS,
    EpochBasedRunner,
    IterBasedRunner,
    IterLoader,
    get_dist_info,
)
from mmcv.runner.utils import get_host_info
from torch.utils.data.dataloader import DataLoader


# pylint: disable=too-many-instance-attributes, attribute-defined-outside-init
@RUNNERS.register_module()
class EpochRunnerWithCancel(EpochBasedRunner):
    """Simple modification to EpochBasedRunner to allow cancelling the training during an epoch.

    A stopping hook should set the runner.should_stop flag to True if stopping is required.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_stop = False
        _, world_size = get_dist_info()
        self.distributed = world_size > 1
        self.save_ema_model = False

    def stop(self) -> bool:
        """Returning a boolean to break the training loop.

        This method supports distributed training by broadcasting should_stop to other ranks
        :return: a cancellation bool
        """
        broadcast_obj = [False]
        if self.rank == 0 and self.should_stop:
            broadcast_obj = [True]

        if self.distributed:
            dist.broadcast_object_list(broadcast_obj, src=0)
        if broadcast_obj[0]:
            self._max_epochs = self.epoch
        return broadcast_obj[0]

    def train(self, data_loader: DataLoader, **kwargs):
        """Train call hook."""
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        if self.distributed:
            time.sleep(2)  # Prevent possible multi-gpu deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook("before_train_iter")
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook("after_train_iter")
            if self.stop():
                break
            self._iter += 1
            self.save_ema_model = False  # revert ema status before new iter
        self.call_hook("after_train_epoch")
        self.stop()
        self._epoch += 1


@RUNNERS.register_module()
class IterBasedRunnerWithCancel(IterBasedRunner):
    """Runner With Cancel for early-stopping (Iter based).

    Simple modification to IterBasedRunner to allow cancelling the training. The cancel training hook
    should set the runner.should_stop flag to True if stopping is required.

    # TODO: Implement cancelling of training via keyboard interrupt signal, instead of should_stop
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_stop = False

    def main_loop(self, workflow: List[tuple], iter_loaders: Sequence[IterLoader], **kwargs):
        """Main loop function in IterBasedRunnerWithCancel."""
        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(f'runner has no method named "{mode}" to run a workflow')
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == "train" and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)
                    if self.should_stop:
                        return

    def run(self, data_loaders: Sequence[DataLoader], workflow: List[tuple], max_iters: Optional[int] = None, **kwargs):
        """Function of main run."""
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                "setting max_iters in run is deprecated, please set max_iters in runner_config",
                DeprecationWarning,
            )
            self._max_iters = max_iters
        assert self._max_iters is not None, "max_iters must be specified during instantiation"

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info("Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.info("workflow: %s, max: %d iters", workflow, self._max_iters)
        self.call_hook("before_run")

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook("before_epoch")

        self.should_stop = False
        self.main_loop(workflow, iter_loaders, **kwargs)
        self.should_stop = False

        # time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_epoch")
        self.call_hook("after_run")
