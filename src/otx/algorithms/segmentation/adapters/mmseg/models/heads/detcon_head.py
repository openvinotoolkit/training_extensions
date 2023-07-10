"""DetCon head to get pixel-wise contrastive loss with predictor head."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=unused-argument
from typing import Any, Dict

import torch
from mmseg.models.builder import HEADS, build_loss, build_neck
from torch import nn


@HEADS.register_module()
class DetConHead(nn.Module):
    """DetCon head for pixel-wise contrastive learning.

    Args:
        predictor (dict): configurations for predictor.
        loss_cfg (dict): configurations for detcon loss.
    """

    def __init__(self, predictor: Dict[str, Any], loss_cfg: Dict[str, Any], **kwargs):
        super().__init__()
        self.predictor = build_neck(predictor)
        self.detcon_loss = build_loss(loss_cfg)

    def init_weights(self):
        """Initialize predictor weights."""
        self.predictor.init_weights()

    # pylint: disable=too-many-locals
    def forward(
        self,
        projs: torch.Tensor,
        projs_tgt: torch.Tensor,
        ids: torch.Tensor,
        ids_tgt: torch.Tensor,
        batch_size: int,
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward head.

        Args:
            projs (Tensor): NxC input features.
            projs_tgt (Tensor): NxC target features.
            ids (Tensor): NxC input ids.
            ids_tgt (Tensor): NxC target ids.
            batch_size (int): Batch size to split concatenated features.
            num_samples (int): The number of samples to be sampled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        preds = self.predictor(projs)
        pred1, pred2 = torch.split(preds.reshape((-1, num_samples, preds.shape[-1])), batch_size)
        id1, id2 = torch.split(ids, batch_size)
        proj1_tgt, proj2_tgt = torch.split(projs_tgt.reshape((-1, num_samples, projs_tgt.shape[-1])), batch_size)
        id1_tgt, id2_tgt = torch.split(ids_tgt, batch_size)

        loss = self.detcon_loss(
            pred1=pred1,
            pred2=pred2,
            target1=proj1_tgt,
            target2=proj2_tgt,
            pind1=id1,
            pind2=id2,
            tind1=id1_tgt,
            tind2=id2_tgt,
        )

        return dict(loss=loss)
