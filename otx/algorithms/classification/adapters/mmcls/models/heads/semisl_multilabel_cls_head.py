"""Module for defining semi-supervised classification head for multi-label classification task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import HEADS, build_loss
from torch import nn

from otx.algorithms.classification.adapters.mmcls.models.heads.custom_multi_label_linear_cls_head import (
    CustomMultiLabelLinearClsHead,
)
from otx.algorithms.classification.adapters.mmcls.models.heads.custom_multi_label_non_linear_cls_head import (
    CustomMultiLabelNonLinearClsHead,
)


def generate_aux_mlp(aux_mlp_cfg: dict, in_channels: int):
    """Generate auxiliary MLP."""
    out_channels = aux_mlp_cfg["out_channels"]
    if out_channels <= 0:
        raise ValueError(f"out_channels={out_channels} must be a positive integer")
    if "hid_channels" in aux_mlp_cfg and aux_mlp_cfg["hid_channels"] > 0:
        hid_channels = aux_mlp_cfg["hid_channels"]
        mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hid_channels, out_features=out_channels),
        )
    else:
        mlp = nn.Linear(in_features=in_channels, out_features=out_channels)

    return mlp


class EMAMeter:
    """EMAMeter class for flexible EMA implementation."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.val = 0

    def reset(self):
        """Reset EMA value."""
        self.val = 0

    def update(self, val):
        """Update ema value."""
        self.val = self.alpha * self.val + (1 - self.alpha) * val


class LossBalancer:
    """LossBalancer class."""

    def __init__(self, num_losses, weights=None, ema_weight=0.7) -> None:
        """Init fuction of LossBalancer class."""
        self.epsilon = 1e-9
        self.avg_estimators = [EMAMeter(ema_weight) for _ in range(num_losses)]

        if weights is not None:
            assert len(weights) == num_losses
            self.final_weights = weights
        else:
            self.final_weights = [1.0] * num_losses

    def balance_losses(self, losses):
        """Balance loss using estimators."""
        total_loss = 0.0
        for i, loss in enumerate(losses):
            self.avg_estimators[i].update(float(loss))
            total_loss += (
                self.final_weights[i] * loss / (self.avg_estimators[i].val + self.epsilon) * self.avg_estimators[0].val
            )

        return total_loss


class SemiMultilabelClsHead:
    """Multilabel Classification head for Semi-SL.

    Args:
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0
        use_dynamic_loss_weighting (boolean): whether to use dynamic unlabeled loss weighting, default is True
    """

    def __init__(
        self,
        unlabeled_coef=0.1,
        use_dynamic_loss_weighting=True,
        aux_loss=dict(type="BarlowTwinsLoss", off_diag_penality=1.0 / 128.0, loss_weight=1.0),
    ):  # pylint: disable=dangerous-default-value
        self.unlabeled_coef = unlabeled_coef
        self.use_dynamic_loss_weighting = use_dynamic_loss_weighting
        self.aux_loss = build_loss(aux_loss)
        if self.use_dynamic_loss_weighting:
            self.loss_balancer = LossBalancer(2, [1.0, unlabeled_coef])
        else:
            self.loss_balancer = None
        self.num_pseudo_label = 0

    def loss(self, logits, gt_label, features):
        """Loss function in which unlabeled data is considered.

        Args:
            logit (Tensor): Labeled data logits
            gt_label (Tensor): target features for labeled data
            features (set): (weak data features, strong data features)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_samples = gt_label.shape[0]
        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        l_labeled = self.compute_loss(
            logits,
            _gt_label,
            avg_factor=num_samples,
        )

        features_weak, features_strong = features
        aux_loss = self.aux_loss(features_weak, features_strong)

        losses = dict(loss=0.0)
        if self.use_dynamic_loss_weighting:
            losses["loss"] = self.loss_balancer.balance_losses((l_labeled, aux_loss))
        else:
            losses["loss"] = l_labeled + self.unlabeled_coef * aux_loss
        losses["unlabeled_loss"] = self.unlabeled_coef * aux_loss

        return losses

    def forward_train_with_last_layers(self, x, gt_label, final_cls_layer, final_emb_layer):
        """Forwards multilabel semi-sl head and losses.

        Args:
            x (dict): dict(labeled_weak. labeled_strong, unlabeled_weak, unlabeled_strong) or NxC input features.
            gt_label (Tensor): NxC target features.
            final_cls_layer (nn.Linear or nn.Sequential): a final layer forwards feature from backbone.
            final_emb_layer (nn.Linear or nn.Sequential): a final layer forwards embeddings from backbone.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        logits = final_cls_layer(x["labeled_weak"])
        features_weak = torch.cat((final_emb_layer(x["labeled_weak"]), final_emb_layer(x["unlabeled_weak"])))
        features_strong = torch.cat((final_emb_layer(x["labeled_strong"]), final_emb_layer(x["unlabeled_strong"])))
        features = (features_weak, features_strong)
        losses = self.loss(logits, gt_label, features)
        return losses


@HEADS.register_module()
class SemiLinearMultilabelClsHead(SemiMultilabelClsHead, CustomMultiLabelLinearClsHead):
    """Linear multilabel classification head for Semi-SL.

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from classifier
        scale (float): Scale for metric learning loss
        normalized (boolean): flag that enables metric learining in loss,
        aux_mlp (dict): Config for embeddings MLP
        loss (dict): configuration of loss, default is CrossEntropyLoss
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0
        use_dynamic_loss_weighting (boolean): whether to use dynamic unlabeled loss weighting, default is True
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        scale=1.0,
        normalized=False,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        unlabeled_coef=0.1,
        aux_loss=dict(type="BarlowTwinsLoss", off_diag_penality=1.0 / 128.0, loss_weight=1.0),
        use_dynamic_loss_weighting=True,
    ):  # pylint: disable=too-many-arguments, dangerous-default-value
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        CustomMultiLabelLinearClsHead.__init__(self, num_classes, in_channels, normalized, scale, loss)
        SemiMultilabelClsHead.__init__(self, unlabeled_coef, use_dynamic_loss_weighting, aux_loss)

        self.aux_mlp = generate_aux_mlp(aux_mlp, in_channels)

    def loss(self, logits, gt_label, features):
        """Calculate loss for given logits/gt_label."""
        return SemiMultilabelClsHead.loss(self, logits, gt_label, features)

    def forward(self, x):
        """Forward fuction of SemiLinearMultilabelClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label):
        """Forward_train fuction of SemiLinearMultilabelClsHead class."""
        return self.forward_train_with_last_layers(
            cls_score, gt_label, final_cls_layer=self.fc, final_emb_layer=self.aux_mlp
        )


@HEADS.register_module()
class SemiNonLinearMultilabelClsHead(SemiMultilabelClsHead, CustomMultiLabelNonLinearClsHead):
    """Non-linear classification head for Semi-SL.

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from classifier
        hid_channels (int): Number of channels of hidden layer.
        scale (float): Scale for metric learning loss
        normalized (boolean): flag that enables metric learining in loss,
        aux_mlp (dict): Config for embeddings MLP
        act_cfg (dict): Config of activation layer
        loss (dict): configuration of loss, default is CrossEntropyLoss
        topk (set): evaluation topk score, default is (1, )
        unlabeled_coef (float): unlabeled loss coefficient, default is 0.1
        use_dynamic_loss_weighting (boolean): whether to use dynamic unlabeled loss weighting, default is True
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        hid_channels=1280,
        scale=1.0,
        normalized=False,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        act_cfg=dict(type="ReLU"),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        aux_loss=dict(type="BarlowTwinsLoss", off_diag_penality=1.0 / 128.0, loss_weight=1.0),
        dropout=False,
        unlabeled_coef=0.1,
        use_dynamic_loss_weighting=True,
    ):  # pylint: disable=too-many-arguments, dangerous-default-value
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        CustomMultiLabelNonLinearClsHead.__init__(
            self,
            num_classes,
            in_channels,
            hid_channels=hid_channels,
            act_cfg=act_cfg,
            loss=loss,
            dropout=dropout,
            scale=scale,
            normalized=normalized,
        )
        SemiMultilabelClsHead.__init__(self, unlabeled_coef, use_dynamic_loss_weighting, aux_loss)

        self.aux_mlp = generate_aux_mlp(aux_mlp, in_channels)

    def loss(self, logits, gt_label, features):
        """Calculate loss for given logits/gt_label."""
        return SemiMultilabelClsHead.loss(self, logits, gt_label, features)

    def forward(self, x):
        """Forward fuction of SemiNonLinearMultilabelClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label):
        """Forward_train fuction of SemiNonLinearMultilabelClsHead class."""
        return self.forward_train_with_last_layers(
            cls_score, gt_label, final_cls_layer=self.classifier, final_emb_layer=self.aux_mlp
        )
