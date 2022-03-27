# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class CosineAnnealingWithWarmupLR(CosineAnnealingLR):
    """Cosine annealing learning rate scheduler."""
    def __init__(self, optimizer, T_warmup, T_max, eta_min=0, last_epoch=-1, is_warmup=True, verbose=False):
        assert T_warmup < T_max
        self.T_warmup = T_warmup
        self.is_warmup = is_warmup
        super().__init__(optimizer, T_max - T_warmup, eta_min, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.T_warmup and self.is_warmup:
            return [base_lr * self.last_epoch / self.T_warmup
                    for base_lr in self.base_lrs]
        else:
            self.switch_warmup()
            return super().get_lr()

    def switch_warmup(self):
        if self.is_warmup:
            self.last_epoch = -1
            self.is_warmup = False
