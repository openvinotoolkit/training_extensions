# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads.base_head import BaseHead
from torch import nn


@HEADS.register_module()
class SupConHierarchicalClsHead(BaseHead):
    """
    Supervised Contrastive Learning head for Classification using SelfSL
    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from the backbone
        aux_mlp (dict): A dictionary with the out_channels and optionally the
                         hid_channels of the auxiliary MLP head.
        loss (dict): The classification for each level (e.g. CrossEntropyLoss)
        mulilabel_loss (dict): The global classification loss (e.g. AsymmetricLoss)
        aux_loss (dict): The SelfSL loss (e.g. BarlowTwinsLoss)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        aux_mlp,
        loss=dict(type="CrossEntropyLoss", reduction="mean", loss_weight=1.0),
        multilabel_loss=dict(type="AsymmetricLoss", reduction="mean", loss_weight=1.0),
        aux_loss=dict(type="BarlowTwinsLoss", off_diag_penality=1.0 / 128.0, loss_weight=1.0),
        init_cfg=None,
        **kwargs,
    ):
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        # Set up the losses
        self.hierarchical_info = kwargs.pop("hierarchical_info", None)
        assert self.hierarchical_info
        super().__init__(init_cfg=init_cfg)
        self.compute_multilabel_loss = False
        if self.hierarchical_info["num_multilabel_classes"] > 0:
            self.compute_multilabel_loss = build_loss(multilabel_loss)
        self.compute_loss = build_loss(loss)
        self.aux_loss = build_loss(aux_loss)

        # Set up the standard classification head
        self.num_classes = num_classes
        self.classifier = nn.Linear(in_features=in_channels, out_features=self.num_classes)

        # Set up the auxiliar head
        out_channels = aux_mlp["out_channels"]
        if out_channels <= 0:
            raise ValueError(f"out_channels={out_channels} must be a positive integer")
        if "hid_channels" in aux_mlp and aux_mlp["hid_channels"] > 0:
            hid_channels = aux_mlp["hid_channels"]
            self.aux_mlp = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hid_channels, out_features=out_channels),
            )
        else:
            self.aux_mlp = nn.Linear(in_features=in_channels, out_features=out_channels)

    def loss(self, cls_score, gt_label, multilabel=False, valid_label_mask=None):
        num_samples = len(cls_score)
        # compute loss
        if multilabel:
            gt_label = gt_label.type_as(cls_score)
            # map difficult examples to positive ones
            _gt_label = torch.abs(gt_label)

            loss = self.compute_multilabel_loss(
                cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples
            )
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)

        return loss

    def forward_train(self, x, gt_label, **kwargs):
        """
        Forward train head using the Supervised Contrastive Loss
        Args:
            x (Tensor): features from the backbone.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img_metas = kwargs.get("img_metas", False)
        gt_label = gt_label.type_as(x)
        cls_score = self.classifier(x)

        bsz = gt_label.shape[0]
        # make sure we have two views for each label and split them
        losses = dict(loss=0.0)
        assert x.shape[0] == 2 * bsz
        feats1, feats2 = torch.split(self.aux_mlp(x), [bsz, bsz], dim=0)
        gt_label = torch.cat([gt_label, gt_label], dim=0)

        # Compute the classification loss
        for i in range(self.hierarchical_info["num_multiclass_heads"]):
            head_gt = gt_label[:, i]
            head_logits = cls_score[
                :,
                self.hierarchical_info["head_idx_to_logits_range"][i][0] : self.hierarchical_info[
                    "head_idx_to_logits_range"
                ][i][1],
            ]

            valid_mask = head_gt >= 0
            head_gt = head_gt[valid_mask].long()
            head_logits = head_logits[valid_mask, :]
            multiclass_loss = self.loss(head_logits, head_gt)
            losses["loss"] += multiclass_loss

        if self.hierarchical_info["num_multiclass_heads"] > 1:
            losses["loss"] /= self.hierarchical_info["num_multiclass_heads"]

        if self.compute_multilabel_loss:
            head_gt = gt_label[:, self.hierarchical_info["num_multiclass_heads"] :]
            head_logits = cls_score[:, self.hierarchical_info["num_single_label_classes"] :]
            valid_mask = head_gt >= 0
            head_gt = head_gt[valid_mask].view(*valid_mask.shape)
            head_logits = head_logits[valid_mask].view(*valid_mask.shape)

            # multilabel_loss is assumed to perform no batch averaging
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)[
                :, self.hierarchical_info["num_single_label_classes"] :
            ]
            valid_label_mask = torch.cat([valid_label_mask, valid_label_mask], dim=0)
            multilabel_loss = self.loss(head_logits, head_gt, multilabel=True, valid_label_mask=valid_label_mask)
            losses["loss"] += multilabel_loss.mean()
        return losses

        bsz = gt_label.shape[0]
        aux_feats = None
        if x.shape[0] == 2 * bsz:
            # reshape aux_feats from [2 * bsz, dims] to [bs, 2, dims]
            feats1, feats2 = torch.split(self.aux_mlp(x), [bsz, bsz], dim=0)
            aux_feats = torch.cat([feats1.unsqueeze(1), feats2.unsqueeze(1)], dim=1)
        if img_metas:
            num_samples = len(cls_score)
            gt_label = gt_label.type_as(cls_score)
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            valid_label_mask = torch.cat([valid_label_mask, valid_label_mask], dim=0)
            loss = self.compute_loss(
                cls_score, gt_label, aux_feats=aux_feats, valid_label_mask=valid_label_mask, avg_factors=num_samples
            )
        else:
            loss = self.compute_loss(cls_score, gt_label, aux_feats=aux_feats)
        losses.update(loss)
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.classifier(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        multiclass_logits = []
        for i in range(self.hierarchical_info["num_multiclass_heads"]):
            multiclass_logit = cls_score[
                :,
                self.hierarchical_info["head_idx_to_logits_range"][i][0] : self.hierarchical_info[
                    "head_idx_to_logits_range"
                ][i][1],
            ]
            multiclass_logits.append(multiclass_logit)
        multiclass_logits = torch.cat(multiclass_logits, dim=1)
        multiclass_pred = torch.softmax(multiclass_logits, dim=1) if multiclass_logits is not None else None

        if self.compute_multilabel_loss:
            multilabel_logits = cls_score[:, self.hierarchical_info["num_single_label_classes"] :]
            multilabel_pred = torch.sigmoid(multilabel_logits) if multilabel_logits is not None else None
            pred = torch.cat([multiclass_pred, multilabel_pred], axis=1)
        else:
            pred = multiclass_pred

        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            mask = mask.cuda() if torch.cuda.is_available() else mask
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
