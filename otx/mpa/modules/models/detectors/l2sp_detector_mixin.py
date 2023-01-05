# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.modules.models.losses.l2sp_loss import L2SPLoss


class L2SPDetectorMixin(object):
    """L2SP-enabled detector mix-in"""

    def __init__(self, l2sp_ckpt=None, l2sp_weight=None, **kwargs):
        super(L2SPDetectorMixin, self).__init__(**kwargs)
        if l2sp_ckpt and l2sp_weight:
            self.l2sp = L2SPLoss(self, l2sp_ckpt, l2sp_weight)
            print("L2SP initilaized!")
        else:
            self.l2sp = None

    def forward_train(self, *args, **kwargs):
        losses = super().forward_train(*args, **kwargs)

        # Add L2SP regularization loss
        # (Assuming weight decay is disable in optimizer setting)
        if self.l2sp:
            losses.update(dict(loss_l2sp=self.l2sp()))
        return losses
