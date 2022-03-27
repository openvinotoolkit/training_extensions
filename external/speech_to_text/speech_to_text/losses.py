# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """Basic CTC loss for speech to text task."""
    def __init__(self, blank_id: int = 0):
        super().__init__()
        self.criterion = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            pred_lengths: torch.Tensor,
            gt_lengths: torch.Tensor
    ):
        pred = pred.permute(1, 0, 2).log_softmax(dim=-1)
        loss = self.criterion(pred, gt, pred_lengths, gt_lengths)
        return {"loss": loss}
