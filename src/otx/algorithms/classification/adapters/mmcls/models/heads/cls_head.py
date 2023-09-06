"""Module defining Classification Head for MMOV inference."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead as OriginClsHead


@HEADS.register_module(force=True)
class ClsHead(OriginClsHead):
    """Classification Head for MMOV inference."""

    def __init__(self, *args, **kwargs):
        do_squeeze = kwargs.pop("do_squeeze", False)
        super().__init__(*args, **kwargs)
        self._do_squeeze = do_squeeze

    def forward(self, x):
        """Forward fuction of ClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label):
        """Forward_train fuction of ClsHead class."""
        if self._do_squeeze:
            cls_score = cls_score.unsqueeze(0).squeeze()
        return super().forward_train(cls_score, gt_label)

    def simple_test(self, cls_score):
        """Test without augmentation."""
        if self._do_squeeze:
            cls_score = cls_score.unsqueeze(0).squeeze()
        return super().simple_test(cls_score)
