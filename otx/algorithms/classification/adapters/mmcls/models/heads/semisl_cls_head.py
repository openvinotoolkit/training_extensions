"""Module for defining semi-supervised learning for multi-class classification task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import HEADS
from mmcls.models.heads.linear_head import LinearClsHead

from otx.algorithms.classification.adapters.mmcls.models.heads.non_linear_cls_head import (
    NonLinearClsHead,
)


class SemiClsHead:
    """Classification head for Semi-SL.

    Args:
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0
        dynamic_threshold (boolean): whether to use dynamic threshold, default is True
        min_threshold (float): Minimum value of threshold determining pseudo-label, default is 0.5
    """

    def __init__(self, unlabeled_coef=1.0, use_dynamic_threshold=True, min_threshold=0.5):
        self.unlabeled_coef = unlabeled_coef
        self.use_dynamic_threshold = use_dynamic_threshold
        self.min_threshold = (
            min_threshold if self.use_dynamic_threshold else 0.95
        )  # the range of threshold will be [min_thr, 1.0]
        self.num_pseudo_label = 0
        self.classwise_acc = torch.ones((self.num_classes,)) * self.min_threshold
        if torch.cuda.is_available():
            self.classwise_acc = self.classwise_acc.cuda()

    def loss(self, logits, gt_label, pseudo_label=None, mask=None):
        """Loss function in which unlabeled data is considered.

        Args:
            logit (set): (labeled data logit, unlabeled data logit)
            gt_label (Tensor): target features for labeled data
            pseudo_label (Tensor): target feature for unlabeled data
            mask (Tensor): Mask that shows pseudo-label that passes threshold

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        logits_x, logits_u_s = logits
        num_samples = len(logits_x)
        losses = dict()

        # compute supervised loss
        labeled_loss = self.compute_loss(logits_x, gt_label, avg_factor=num_samples)

        unlabeled_loss = 0
        if len(logits_u_s) > 0:
            # compute unsupervised loss
            unlabeled_loss = self.compute_loss(logits_u_s, pseudo_label, avg_factor=len(logits_u_s)) * mask
        losses["loss"] = labeled_loss + self.unlabeled_coef * unlabeled_loss
        losses["unlabeled_loss"] = self.unlabeled_coef * unlabeled_loss

        # compute accuracy
        acc = self.compute_accuracy(logits_x, gt_label)
        losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        return losses

    def forward_train(self, x, gt_label, final_layer=None):  # pylint: disable=too-many-locals
        """Forward_train head using pseudo-label selected through threshold.

        Args:
            x (dict or Tensor): dict(labeled, unlabeled_weak, unlabeled_strong) or NxC input features.
            gt_label (Tensor): NxC target features.
            final_layer (nn.Linear or nn.Sequential): a final layer forwards feature from backbone.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        label_u, mask = None, None
        if isinstance(x, dict):
            outputs = final_layer(x["labeled"])  # Logit of Labeled Img
            batch_size = len(outputs)

            with torch.no_grad():
                logit_uw = final_layer(x["unlabeled_weak"])
                pseudo_label = torch.softmax(logit_uw.detach(), dim=-1)
                max_probs, label_u = torch.max(pseudo_label, dim=-1)

                # select Pseudo-Label using flexible threhold
                if torch.cuda.is_available():
                    max_probs = max_probs.cuda()
                mask = max_probs.ge(self.classwise_acc[label_u]).float()
                self.num_pseudo_label = mask.sum()

                if self.use_dynamic_threshold:
                    # get Labeled Data True Positive Confidence
                    logit_x = torch.softmax(outputs.detach(), dim=-1)
                    x_probs, x_idx = torch.max(logit_x, dim=-1)
                    x_probs = x_probs[x_idx == gt_label]
                    x_idx = x_idx[x_idx == gt_label]

                    # get Unlabeled Data Selected Confidence
                    uw_probs = max_probs[mask == 1]
                    uw_idx = label_u[mask == 1]

                    # update class-wise accuracy
                    for i in set(x_idx.tolist() + uw_idx.tolist()):
                        current_conf = torch.tensor([x_probs[x_idx == i].mean(), uw_probs[uw_idx == i].mean()])
                        current_conf = current_conf[~current_conf.isnan()].mean()
                        self.classwise_acc[i] = max(current_conf, self.min_threshold)

            outputs = torch.cat((outputs, final_layer(x["unlabeled_strong"])))
        else:
            outputs = final_layer(x)
            batch_size = len(outputs)

        logits_x = outputs[:batch_size]
        logits_u = outputs[batch_size:]
        del outputs
        logits = (logits_x, logits_u)
        losses = self.loss(logits, gt_label, label_u, mask)
        return losses


@HEADS.register_module()
class SemiLinearClsHead(SemiClsHead, LinearClsHead):
    """Linear classification head for Semi-SL.

    This head is designed to support FixMatch algorithm. (https://arxiv.org/abs/2001.07685)
        - [OTX] supports dynamic threshold based on confidence for each class

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from classifier
        loss (dict): configuration of loss, default is CrossEntropyLoss
        topk (set): evaluation topk score, default is (1, )
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0
        dynamic_threshold (boolean): whether to use dynamic threshold, default is True
        min_threshold (float): Minimum value of threshold determining pseudo-label, default is 0.5
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        loss=None,
        topk=None,
        unlabeled_coef=1.0,
        use_dynamic_threshold=True,
        min_threshold=0.5,
    ):  # pylint: disable=too-many-arguments
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        topk = (1,) if num_classes < 5 else (1, 5)
        loss = loss if loss else dict(type="CrossEntropyLoss", loss_weight=1.0)
        LinearClsHead.__init__(self, num_classes, in_channels, loss=loss, topk=topk)
        SemiClsHead.__init__(self, unlabeled_coef, use_dynamic_threshold, min_threshold)

    def forward(self, x):
        """Forward fuction of SemiLinearClsHead class."""
        return self.simple_test(x)

    def forward_train(self, x, gt_label):
        """Forward_train fuction of SemiLinearClsHead class."""
        return SemiClsHead.forward_train(self, x, gt_label, final_layer=self.fc)


@HEADS.register_module()
class SemiNonLinearClsHead(SemiClsHead, NonLinearClsHead):
    """Non-linear classification head for Semi-SL.

    This head is designed to support FixMatch algorithm. (https://arxiv.org/abs/2001.07685)
        - [OTX] supports dynamic threshold based on confidence for each class

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from classifier
        hid_channels (int): Number of channels of hidden layer.
        act_cfg (dict): Config of activation layer.
        loss (dict): configuration of loss, default is CrossEntropyLoss
        topk (set): evaluation topk score, default is (1, )
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0
        dynamic_threshold (boolean): whether to use dynamic threshold, default is True
        min_threshold (float): Minimum value of threshold determining pseudo-label, default is 0.5
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        hid_channels=1280,
        act_cfg=None,
        loss=None,
        topk=None,
        dropout=False,
        unlabeled_coef=1.0,
        use_dynamic_threshold=True,
        min_threshold=0.5,
    ):  # pylint: disable=too-many-arguments
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        topk = (1,) if num_classes < 5 else (1, 5)
        act_cfg = act_cfg if act_cfg else dict(type="ReLU")
        loss = loss if loss else dict(type="CrossEntropyLoss", loss_weight=1.0)
        NonLinearClsHead.__init__(
            self,
            num_classes,
            in_channels,
            hid_channels=hid_channels,
            act_cfg=act_cfg,
            loss=loss,
            topk=topk,
            dropout=dropout,
        )
        SemiClsHead.__init__(self, unlabeled_coef, use_dynamic_threshold, min_threshold)

    def forward(self, x):
        """Forward fuction of SemiNonLinearClsHead class."""
        return self.simple_test(x)

    def forward_train(self, x, gt_label):
        """Forward_train fuction of SemiNonLinearClsHead class."""
        return SemiClsHead.forward_train(self, x, gt_label, final_layer=self.classifier)
