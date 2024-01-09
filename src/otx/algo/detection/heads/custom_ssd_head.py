# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom SSD head for OTX template."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.registry import MODELS
from torch import nn

if TYPE_CHECKING:
    from mmengine.config import Config


@MODELS.register_module()
class CustomSSDHead(SSDHead):
    """CustomSSDHead class for OTX.

    This is workaround for bug in mmdet3.2.0
    """

    def __init__(self, *args, loss_cls: Config | dict | None = None, **kwargs) -> None:
        """Initialize CustomSSDHead."""
        super().__init__(*args, **kwargs)
        if loss_cls is None:
            loss_cls = {
                "type": "CrossEntropyLoss",
                "use_sigmoid": False,
                "reduction": "none",
                "loss_weight": 1.0,
            }
        self.loss_cls = MODELS.build(loss_cls)

    def _init_layers(self) -> None:
        """Initialize layers of the head.

        This modificaiton is needed for smart weight loading
        """
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        act_cfg = self.act_cfg.copy()
        act_cfg.setdefault("inplace", True)
        for in_channel, num_base_priors in zip(self.in_channels, self.num_base_priors):
            if self.use_depthwise:
                activation_layer = MODELS.build(act_cfg)

                self.reg_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=1, padding=0),
                    ),
                )
                self.cls_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=1, padding=0),
                    ),
                )
            else:
                self.reg_convs.append(nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=3, padding=1))
                self.cls_convs.append(
                    nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=3, padding=1),
                )
