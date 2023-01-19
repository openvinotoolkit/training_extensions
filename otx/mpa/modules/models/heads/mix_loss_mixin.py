# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32


class MixLossMixin(nn.Module):
    @staticmethod
    def _mix_loss(logits, target, ignore_index=255):
        num_samples = logits.size(0)
        assert num_samples % 2 == 0

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            probs_a, probs_b = torch.split(probs, num_samples // 2)
            mean_probs = 0.5 * (probs_a + probs_b)
            trg_probs = torch.cat([mean_probs, mean_probs], dim=0)

        log_probs = torch.log_softmax(logits, dim=1)
        losses = torch.sum(trg_probs * log_probs, dim=1).neg()

        valid_mask = target != ignore_index
        valid_losses = torch.where(valid_mask, losses, torch.zeros_like(losses))

        return valid_losses.mean()

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, train_cfg, *args, **kwargs):
        loss = super().losses(seg_logit, seg_label, train_cfg, *args, **kwargs)
        if train_cfg.get("mix_loss", None) and train_cfg.mix_loss.get("enable", False):
            mix_loss = self._mix_loss(seg_logit, seg_label, ignore_index=self.ignore_index)

            mix_loss_weight = train_cfg.mix_loss.get("weight", 1.0)
            loss["loss_mix"] = mix_loss_weight * mix_loss

        return loss
