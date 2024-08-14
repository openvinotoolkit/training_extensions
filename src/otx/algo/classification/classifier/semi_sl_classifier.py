# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Semi-SL Classifier Implementation.

Implementing Classifier with dict logic to handle unlabeled datasets with multi-transforms

"""

from __future__ import annotations

import torch
from torch import nn

from otx.algo.classification.heads.semi_sl_head import OTXSemiSLClsHead

from .base_classifier import ImageClassifier


class SemiSLClassifier(ImageClassifier):
    """Semi-SL classifier.

    Args:
        backbone (nn.Module): The backbone network.
        neck (nn.Module | None): The neck module. Defaults to None.
        head (nn.Module): The head module.
        loss (nn.Module): The loss module.
        unlabeled_coef (float): The coefficient for the unlabeled loss. Defaults to 1.0.
        init_cfg (dict | list[dict] | None): The initialization configuration. Defaults to None.
    """

    head: OTXSemiSLClsHead

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        loss: nn.Module,
        unlabeled_coef: float = 1.0,
        init_cfg: dict | list[dict] | None = None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=loss,
            init_cfg=init_cfg,
        )
        self.unlabeled_coef = unlabeled_coef

    def extract_feat(
        self,
        inputs: dict[str, torch.Tensor] | torch.Tensor,
        stage: str = "neck",
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor] | torch.Tensor:
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (dict[str, torch.Tensor] | torch.Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.

                Defaults to "neck".

        Returns:
            dict[str, torch.Tensor] | tuple[torch.Tensor] | torch.Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the Semi-SL
            output is a dict of labeled feats and unlabeled feats.
        """
        if not isinstance(inputs, dict):
            return super().extract_feat(inputs, stage)

        labeled_inputs = inputs["labeled"]
        unlabeled_weak_inputs = inputs["weak_transforms"]
        unlabeled_strong_inputs = inputs["strong_transforms"]

        x = {}
        x["labeled"] = super().extract_feat(labeled_inputs, stage)
        # For weak augmentation inputs, use no_grad to use as a pseudo-label.
        with torch.no_grad():
            x["unlabeled_weak"] = super().extract_feat(unlabeled_weak_inputs, stage)
        x["unlabeled_strong"] = super().extract_feat(unlabeled_strong_inputs, stage)

        return x

    def loss(self, inputs: dict[str, torch.Tensor], labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            labels (torch.Tensor): The annotation data of
                every samples.

        Returns:
            torch.Tensor: loss components
        """
        semi_inputs = self.extract_feat(inputs)
        logits, labels, pseudo_label, mask = self.head.get_logits(semi_inputs, labels)
        logits_x, logits_u_s = logits
        num_samples = len(logits_x)

        # compute supervised loss
        labeled_loss = self.loss_module(logits_x, labels).sum() / num_samples

        unlabeled_loss = torch.tensor(0.0)
        num_pseudo_labels = 0 if mask is None else int(mask.sum().item())
        if len(logits_u_s) > 0 and num_pseudo_labels > 0 and mask is not None:
            # compute unsupervised loss
            unlabeled_loss = (self.loss_module(logits_u_s, pseudo_label) * mask).sum() / mask.sum().item()
            unlabeled_loss.masked_fill_(torch.isnan(unlabeled_loss), 0.0)

        return labeled_loss + self.unlabeled_coef * unlabeled_loss
