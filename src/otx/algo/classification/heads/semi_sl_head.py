# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Semi-SL for OTX Head Implementation."""


from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from otx.algo.utils.weight_init import constant_init, normal_init

from .linear_head import LinearClsHead


class OTXSemiSLClsHead(nn.Module):
    """Semi-SL for OTX Head Implementation.

    This implementation is based on the algorithmic papers below, with new ideas and modifications.

    This implementation is based on the algorithmic papers below, with new ideas and modifications.
        - FixMatch (2020): https://arxiv.org/abs/2001.07685
        - AdaMatch (2021): https://arxiv.org/abs/2106.04732
        - FlexMatch (2022): https://arxiv.org/abs/2110.08263

    Methods:
        - Pseudo-Labeling (Self-training) with unlabeled sample (2013)
        - Consistency Regularization (2016)
        - Dynamic Threshold
            - Related Confidence Threshold (2022)
            - Per-Class Threshold
    """

    loss_module: nn.Module

    def __init__(
        self,
        num_classes: int,
        unlabeled_coef: float = 1.0,
        use_dynamic_threshold: bool = True,
        min_threshold: float = 0.5,
    ):
        """Initializes the OTXSemiSLClsHead class.

        Args:
            num_classes (int): The number of classes.
            unlabeled_coef (float, optional): The coefficient for the unlabeled loss. Defaults to 1.0.
            use_dynamic_threshold (bool, optional): Whether to use a dynamic threshold for pseudo-label selection.
                Defaults to True.
            min_threshold (float, optional): The minimum threshold for pseudo-label selection. Defaults to 0.5.
        """
        self.num_classes = num_classes

        self.unlabeled_coef = unlabeled_coef
        self.use_dynamic_threshold = use_dynamic_threshold
        self.min_threshold = (
            min_threshold if self.use_dynamic_threshold else 0.95
        )  # the range of threshold will be [min_thr, 1.0]
        self.num_pseudo_label = 0
        self.classwise_acc = torch.ones((self.num_classes,)) * self.min_threshold

    def loss(
        self,
        feats: dict[str, torch.Tensor] | tuple[torch.Tensor] | torch.Tensor,
        labels: dict[str, torch.Tensor] | torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes the loss function in which unlabeled data is considered.

        Args:
            feats (dict or Tensor): Input features.
            labels (Tensor): Target features.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The computed loss.
        """
        logits, labels, pseudo_label, mask = self.get_logits(feats, labels)
        logits_x, logits_u_s = logits
        num_samples = len(logits_x)

        # compute supervised loss
        labeled_loss = self.loss_module(logits_x, labels).sum() / num_samples

        unlabeled_loss = torch.tensor(0.0)
        if len(logits_u_s) > 0 and self.num_pseudo_label > 0 and mask is not None:
            # compute unsupervised loss
            unlabeled_loss = (self.loss_module(logits_u_s, pseudo_label) * mask).sum() / mask.sum().item()
            unlabeled_loss.masked_fill_(torch.isnan(unlabeled_loss), 0.0)

        return labeled_loss + self.unlabeled_coef * unlabeled_loss

    def get_logits(
        self,
        feats: dict[str, torch.Tensor] | tuple[torch.Tensor] | torch.Tensor,
        labels: dict[str, torch.Tensor] | torch.Tensor,
    ) -> tuple[tuple[Any, Any], torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward_train head using pseudo-label selected through threshold.

        Args:
            feats (dict or Tensor): Input features.
            labels (Tensor): Target features.

        Returns:
            tuple: A tuple containing the logits, labels, pseudo-labels, and mask.
        """
        if isinstance(labels, dict):
            labels = labels["labeled"]
        label_u, mask = None, None
        if isinstance(feats, dict):
            for key in feats:
                if isinstance(feats[key], list):
                    feats[key] = feats[key][-1]
            outputs = self(feats["labeled"])  # Logit of Labeled Img
            batch_size = len(outputs)

            with torch.no_grad():
                logit_uw = self(feats["unlabeled_weak"])
                pseudo_label = torch.softmax(logit_uw.detach(), dim=-1)
                max_probs, label_u = torch.max(pseudo_label, dim=-1)

                # select Pseudo-Label using flexible threhold
                self.classwise_acc = self.classwise_acc.to(label_u.device)
                mask = max_probs.ge(self.classwise_acc[label_u]).float()
                self.num_pseudo_label = int(mask.sum().item())

                if self.use_dynamic_threshold:
                    # get Labeled Data True Positive Confidence
                    logit_x = torch.softmax(outputs.detach(), dim=-1)
                    x_probs, x_idx = torch.max(logit_x, dim=-1)
                    x_probs = x_probs[x_idx == labels]
                    x_idx = x_idx[x_idx == labels]

                    # get Unlabeled Data Selected Confidence
                    uw_probs = max_probs[mask == 1]
                    uw_idx = label_u[mask == 1]

                    # update class-wise accuracy
                    for i in set(x_idx.tolist() + uw_idx.tolist()):
                        current_conf = torch.tensor([x_probs[x_idx == i].mean(), uw_probs[uw_idx == i].mean()])
                        current_conf = current_conf[~current_conf.isnan()].mean()
                        self.classwise_acc[i] = max(float(current_conf), self.min_threshold)

            outputs = torch.cat((outputs, self(feats["unlabeled_strong"])))
        else:
            outputs = self(feats)
            batch_size = len(outputs)

        logits_x = outputs[:batch_size]
        logits_u = outputs[batch_size:]
        del outputs
        logits = (logits_x, logits_u)
        return logits, labels, label_u, mask


class OTXSemiSLLinearClsHead(OTXSemiSLClsHead, LinearClsHead):
    """LinearClsHead for OTXSemiSLClsHead."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        topk: int | tuple = (1,),
        unlabeled_coef: float = 1,
        use_dynamic_threshold: bool = True,
        min_threshold: float = 0.5,
    ):
        LinearClsHead.__init__(self, num_classes=num_classes, in_channels=in_channels, loss=loss, topk=topk)
        OTXSemiSLClsHead.__init__(
            self,
            num_classes=num_classes,
            unlabeled_coef=unlabeled_coef,
            use_dynamic_threshold=use_dynamic_threshold,
            min_threshold=min_threshold,
        )


class OTXSemiSLNonLinearClsHead(OTXSemiSLClsHead):
    """NonLinearClsHead for OTXSemiSLClsHead."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        hid_channels: int = 1280,
        topk: int | tuple = (1,),
        unlabeled_coef: float = 1,
        use_dynamic_threshold: bool = True,
        min_threshold: float = 0.5,
    ):
        self.num_classes = num_classes
        self.loss_module = loss
        self.topk = topk

        self.in_channels = in_channels
        self.hid_channels = hid_channels

        self.classifier = nn.Sequential(
            nn.Linear(self.in_channels, self.hid_channels),
            nn.BatchNorm1d(self.hid_channels),
            self.act,
            nn.Linear(self.hid_channels, self.num_classes),
        )

        OTXSemiSLClsHead.__init__(
            self,
            num_classes=num_classes,
            unlabeled_coef=unlabeled_coef,
            use_dynamic_threshold=use_dynamic_threshold,
            min_threshold=min_threshold,
        )

    def init_weights(self) -> None:
        """Initialize weights of head."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The forward process."""
        if isinstance(feats, Sequence):
            feats = feats[-1]
        return self.classifier(feats)
