# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""RTMDet detector."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .single_stage import InstSegSingleStageDetector

if TYPE_CHECKING:
    import torch
    from torch import nn


class RTMDet(InstSegSingleStageDetector):
    """Implementation of RTMDet.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
        use_syncbn (bool): Whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        bbox_head: nn.Module,
        data_preprocessor: nn.Module,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

    def export(self, batch_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Export the model."""
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.export(x)
