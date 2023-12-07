# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom SSD head for OTX template."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.registry import MODELS

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
