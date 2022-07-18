# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Is based on
# * https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/epoch_based_runner.py
# * https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py

import time

import torch.distributed as dist
from mmcv.runner import RUNNERS, EpochBasedRunner, get_dist_info
from torch.utils.data.dataloader import DataLoader

from ote_sdk.utils.argument_checks import check_input_parameters_type


@RUNNERS.register_module()
class EpochRunnerWithCancel(EpochBasedRunner):
    """
    Simple modification to EpochBasedRunner to allow cancelling the training during an epoch.
    A stopping hook should set the runner.should_stop flag to True if stopping is required.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_stop = False
        _, world_size = get_dist_info()
        self.distributed = True if world_size > 1 else False

    def stop(self) -> bool:
        """ Returning a boolean to break the training loop
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

    @check_input_parameters_type()
    def train(self, data_loader: DataLoader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            if self.stop():
                break
            self._iter += 1
        self.call_hook('after_train_epoch')
        self.stop()
        self._epoch += 1
