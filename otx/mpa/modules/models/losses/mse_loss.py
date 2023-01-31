# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.builder import LOSSES
from torch.nn import MSELoss as PytorchMSELoss


@LOSSES.register_module()
class MSELoss(PytorchMSELoss):
    def __init__(self, loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, **kwargs):
        return self.loss_weight * super().forward(cls_score, label)
