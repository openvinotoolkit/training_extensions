# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.atss_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/atss_head.py
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn

from otx.algo.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss
from otx.algo.common.utils.bbox_overlaps import bbox_overlaps
from otx.algo.common.utils.utils import multi_apply, reduce_mean
from otx.algo.detection.heads.anchor_head import AnchorHead
from otx.algo.detection.heads.class_incremental_mixin import (
    ClassIncrementalMixin,
)
from otx.algo.detection.utils.prior_generators.utils import anchor_inside_flags
from otx.algo.detection.utils.utils import unmap
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer
from otx.algo.modules.scale import Scale
from otx.algo.utils.mmengine_utils import InstanceData

EPS = 1e-12


class ATSSHead(ClassIncrementalMixin, AnchorHead):
    """Detection Head of `ATSS <https://arxiv.org/abs/1912.02424>`_.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        pred_kernel_size (int): Kernel size of ``nn.Conv2d``. Defaults to 3.
        stacked_convs (int): Number of stacking convs of the head. Defaults to 4.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``partial(build_norm_layer, nn.GroupNorm, num_groups=32, requires_grad=True)``.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_centerness (nn.Module, optinoal): Module of centerness loss. Defaults to None.
        init_cfg (dict, list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        pred_kernel_size: int = 3,
        stacked_convs: int = 4,
        normalization: Callable[..., nn.Module] = partial(
            build_norm_layer,
            nn.GroupNorm,
            num_groups=32,
            requires_grad=True,
        ),
        reg_decoded_bbox: bool = True,
        loss_centerness: nn.Module | None = None,
        init_cfg: dict | None = None,
        bg_loss_weight: float = -1.0,
        use_qfl: bool = False,
        qfl_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.normalization = normalization
        init_cfg = init_cfg or {
            "type": "Normal",
            "layer": "Conv2d",
            "std": 0.01,
            "override": {"type": "Normal", "name": "atss_cls", "std": 0.01, "bias_prob": 0.01},
        }
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.sampling = False
        self.loss_centerness = loss_centerness or CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)

        if use_qfl:
            kwargs["loss_cls"] = (
                qfl_cfg
                if qfl_cfg
                else {
                    "type": "QualityFocalLoss",
                    "use_sigmoid": True,
                    "beta": 2.0,
                    "loss_weight": 1.0,
                }
            )
        self.bg_loss_weight = bg_loss_weight
        self.use_qfl = use_qfl

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                Conv2dModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                ),
            )
            self.reg_convs.append(
                Conv2dModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                ),
            )
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, x: tuple[Tensor]) -> tuple[list[Tensor], ...]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, x, self.scales)

    def forward_single(self, x: Tensor, scale: Scale) -> tuple[Tensor, ...]:  # type: ignore[override]
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (Scale): Learnable scale module to resize the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat))
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_by_feat(  # type: ignore[override]
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        centernesses: list[Tensor],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            batch_gt_instances (list[InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        if len(featmap_sizes) != self.prior_generator.num_levels:
            msg = "featmap_sizes and self.prior_generator.num_levels have different levels."
            raise ValueError(msg)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor,
            valid_label_mask,
        ) = cls_reg_targets
        avg_factor = reduce_mean(torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            valid_label_mask,
            avg_factor=avg_factor,
        )

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = [loss_bbox / bbox_avg_factor for loss_bbox in losses_bbox]
        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox, "loss_centerness": loss_centerness}

    def loss_by_feat_single(  # type: ignore[override]
        self,
        anchors: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        centerness: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        valid_label_mask: Tensor,
        avg_factor: float,
    ) -> tuple:
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centerness(Tensor): Centerness scores for each scale level.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            valid_label_mask (Tensor): Label mask for consideration of ignored
                label with shape (N, num_total_anchors, 1).
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            tuple[Tensor]: A tuple of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        valid_label_mask = valid_label_mask.reshape(-1, self.cls_out_channels)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = self._get_pos_inds(labels)

        if self.use_qfl:
            quality = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            if self.reg_decoded_bbox:
                pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)

            if self.use_qfl:
                quality[pos_inds] = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_targets, is_aligned=True).clamp(
                    min=1e-6,
                )

            # regression loss
            loss_bbox = self._get_loss_bbox(pos_bbox_targets, pos_bbox_pred, centerness_targets)

            # centerness loss
            loss_centerness = self._get_loss_centerness(avg_factor, pos_centerness, centerness_targets)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.0)

        # Re-weigting BG loss
        if self.bg_loss_weight >= 0.0:
            neg_indices = labels == self.num_classes
            label_weights[neg_indices] = self.bg_loss_weight

        if self.use_qfl:
            labels = (labels, quality)  # For quality focal loss arg spec

        # classification loss
        loss_cls = self._get_loss_cls(cls_score, labels, label_weights, valid_label_mask, avg_factor)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def centerness_target(self, anchors: Tensor, gts: Tensor) -> Tensor:
        """Calculate the centerness between anchors and gts.

        Only calculate pos centerness targets, otherwise there may be nan.

        Args:
            anchors (Tensor): Anchors with shape (N, 4), "xyxy" format.
            gts (Tensor): Ground truth bboxes with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Centerness between anchors and gts.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        return torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
            * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]),
        )

    def _get_pos_inds(self, labels: Tensor) -> Tensor:
        bg_class_ind = self.num_classes
        return ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

    def _get_loss_cls(
        self,
        cls_score: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        valid_label_mask: Tensor,
        avg_factor: Tensor,
    ) -> Tensor:
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                valid_label_mask=valid_label_mask,
            )
        else:
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)
        return loss_cls

    def _get_loss_centerness(
        self,
        avg_factor: Tensor,
        pos_centerness: Tensor,
        centerness_targets: Tensor,
    ) -> Tensor:
        return self.loss_centerness(pos_centerness, centerness_targets, avg_factor=avg_factor)

    def _get_loss_bbox(
        self,
        pos_bbox_targets: Tensor,
        pos_bbox_pred: Tensor,
        centerness_targets: Tensor,
    ) -> Tensor:
        return self.loss_bbox(pos_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0)

    def get_targets(
        self,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Get targets for Detection head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        However, if the detector's head loss uses CrossSigmoidFocalLoss,
        the labels_weights_list consists of (binarized label schema * weights) of batch images
        """
        return self.get_atss_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs,
        )

    def _get_targets_single(  # type: ignore[override]
        self,
        flat_anchors: Tensor,
        valid_flags: Tensor,
        num_level_anchors: list[int],
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: InstanceData | None = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Compute regression, classification targets for anchors in a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors (List[int]): Number of anchors of each scale
                level.
            gt_instances (InstanceData): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (InstanceData, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
                sampling_result (`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.train_cfg["allowed_border"],
        )
        if not inside_flags.any():
            msg = (
                "There is no valid anchor inside the image boundary. Please "
                "check the image size and anchor sizes, or set "
                "``allowed_border`` to -1 to skip the condition.",
            )
            raise ValueError(msg)
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(  # type: ignore[call-arg]
            pred_instances,
            num_level_anchors_inside,  # type: ignore[arg-type]
            gt_instances,
            gt_instances_ignore,
        )

        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg["pos_weight"] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg["pos_weight"]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)

    def get_num_level_anchors_inside(self, num_level_anchors: list[int], inside_flags: Tensor) -> list[int]:
        """Get the number of valid anchors in every level."""
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        return [int(flags.sum()) for flags in split_inside_flags]
