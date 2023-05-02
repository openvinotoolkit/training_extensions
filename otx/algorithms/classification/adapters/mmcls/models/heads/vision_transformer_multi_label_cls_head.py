"""Module for defining multi-label classification head for vision transformer models."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import HEADS
from mmcls.models.heads import VisionTransformerClsHead


@HEADS.register_module()
class VisionTransformerMultiLabelClsHead(VisionTransformerClsHead):
    """Multi-label classification head for VisionTransformer models.

    # TODO: update docstring
    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        """Calculate loss for given cls_score/gt_label."""
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        losses["loss"] = loss
        return losses

    def forward(self, x):
        """Forward fuction of VisionTransformerMultiLabelClsHead class."""
        return self.simple_test(x)

    def forward_train(self, x, gt_label, **kwargs):
        """Forward_train fuction of VisionTransformerMultiLabelClsHead."""
        img_metas = kwargs.get("img_metas", False)
        x = self.pre_logits(x)
        gt_label = gt_label.type_as(x)
        cls_score = self.layers.head(x)

        valid_batch_mask = gt_label >= 0
        gt_label = gt_label[
            valid_batch_mask,
        ].view(gt_label.shape[0], -1)
        cls_score = cls_score[
            valid_batch_mask,
        ].view(cls_score.shape[0], -1)
        if img_metas:
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            valid_label_mask = valid_label_mask.to(cls_score.device)
            valid_label_mask = valid_label_mask[
                valid_batch_mask,
            ].view(valid_label_mask.shape[0], -1)
            losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
        else:
            losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, x):
        """Test without augmentation."""
        x = self.pre_logits(x)
        cls_score = self.layers.head(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if torch.onnx.is_in_onnx_export():
            return cls_score
        pred = torch.sigmoid(cls_score) if cls_score is not None else None
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        """Get valid label mask using ignored_label."""
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
