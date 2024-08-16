# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""SSD criterion."""

from __future__ import annotations

from torch import Tensor, nn

from otx.algo.common.losses import smooth_l1_loss


class SSDCriterion(nn.Module):
    """SSDCriterion is a loss criterion for Single Shot MultiBox Detector (SSD).

    Args:
        num_classes (int): Number of classes including the background class.
        bbox_coder (nn.Module): Bounding box coder module. Defaults to None.
        neg_pos_ratio (int, optional): Ratio of negative to positive samples. Defaults to 3.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        smoothl1_beta (float, optional): Beta parameter for the smooth L1 loss. Defaults to 1.0.
    """

    def __init__(
        self,
        num_classes: int,
        bbox_coder: nn.Module | None = None,
        neg_pos_ratio: int = 3,
        reg_decoded_bbox: bool = False,
        smoothl1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder
        self.neg_pos_ratio = neg_pos_ratio
        self.reg_decoded_bbox = reg_decoded_bbox
        self.smoothl1_beta = smoothl1_beta

    def forward(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        anchor: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        avg_factor: int,
    ) -> dict[str, Tensor]:
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for each image has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor with shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor with shape (num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. the dict
            has components below:

            - loss_cls (list[Tensor]): A list containing each feature map \
            classification loss.
            - loss_bbox (list[Tensor]): A list containing each feature map \
            regression loss.
        """
        loss_cls_all = nn.functional.cross_entropy(cls_score, labels, reduction="none") * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor

        if self.reg_decoded_bbox and self.bbox_coder is not None:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.smoothl1_beta,
            avg_factor=avg_factor,
        )
        return {"loss_cls": loss_cls[None], "loss_bbox": loss_bbox}
