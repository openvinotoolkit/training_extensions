# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import HEADS, build_loss

from otx.mpa.modules.models.heads.custom_multi_label_linear_cls_head import (
    CustomMultiLabelLinearClsHead,
)
from otx.mpa.modules.models.heads.custom_multi_label_non_linear_cls_head import (
    CustomMultiLabelNonLinearClsHead,
)

from .utils import LossBalancer, generate_aux_mlp


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
    ):
        self.unlabeled_coef = unlabeled_coef
        self.use_dynamic_loss_weighting = use_dynamic_loss_weighting
        self.aux_loss = build_loss(aux_loss)
        if self.use_dynamic_loss_weighting:
            self.loss_balancer = LossBalancer(2, [1.0, unlabeled_coef])
        else:
            self.loss_balancer = None
        self.num_pseudo_label = 0

    def loss(self, logits, gt_label, features):
        """loss function in which unlabeled data is considered

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
        """Forwards multilabel semi-sl head and losses

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
    """Linear multilabel classification head for Semi-SL

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
    ):
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        CustomMultiLabelLinearClsHead.__init__(self, num_classes, in_channels, normalized, scale, loss)
        SemiMultilabelClsHead.__init__(self, unlabeled_coef, use_dynamic_loss_weighting, aux_loss)

        self.aux_mlp = generate_aux_mlp(aux_mlp, in_channels)

    def forward_train(self, x, gt_label):
        return self.forward_train_with_last_layers(x, gt_label, final_cls_layer=self.fc, final_emb_layer=self.aux_mlp)


@HEADS.register_module()
class SemiNonLinearMultilabelClsHead(SemiMultilabelClsHead, CustomMultiLabelNonLinearClsHead):
    """Non-linear classification head for Semi-SL

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
    ):
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

    def forward_train(self, x, gt_label):
        return self.forward_train_with_last_layers(
            x, gt_label, final_cls_layer=self.classifier, final_emb_layer=self.aux_mlp
        )
