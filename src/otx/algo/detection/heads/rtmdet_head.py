# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.rtmdet_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/rtmdet_head.py
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn

from otx.algo.common.utils.nms import multiclass_nms
from otx.algo.common.utils.utils import distance2bbox, inverse_sigmoid, multi_apply, reduce_mean
from otx.algo.detection.heads import ATSSHead
from otx.algo.detection.utils.prior_generators.utils import anchor_inside_flags
from otx.algo.detection.utils.utils import (
    images_to_levels,
    sigmoid_geometric_mean,
    unmap,
)
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.modules.norm import build_norm_layer, is_norm
from otx.algo.modules.scale import Scale
from otx.algo.utils.mmengine_utils import InstanceData
from otx.algo.utils.weight_init import bias_init_with_prob, constant_init, normal_init


class RTMDetHead(ATSSHead):
    """Detection Head of RTMDet.

    Args:
        num_classes (int): Number of categories excluding the background category.
        in_channels (int): Number of channels in the input feature map.
        with_objectness (bool): Whether to add an objectness branch.
            Defaults to True.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.ReLU``.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        with_objectness: bool = True,
        activation: Callable[..., nn.Module] = nn.ReLU,
        **kwargs,
    ) -> None:
        self.activation = activation
        self.with_objectness = with_objectness
        super().__init__(num_classes, in_channels, **kwargs)
        if self.train_cfg:
            self.assigner = self.train_cfg["assigner"]

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
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
                    activation=build_activation_layer(self.activation),
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
                    activation=build_activation_layer(self.activation),
                ),
            )
        pred_pad_size = self.pred_kernel_size // 2
        self.rtm_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.rtm_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        if self.with_objectness:
            self.rtm_obj = nn.Conv2d(self.feat_channels, 1, self.pred_kernel_size, padding=pred_pad_size)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.rtm_cls, std=0.01, bias=bias_cls)
        normal_init(self.rtm_reg, std=0.01)
        if self.with_objectness:
            normal_init(self.rtm_obj, std=0.01, bias=bias_cls)

    def forward(self, feats: tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for x, scale, stride in zip(feats, self.scales, self.prior_generator.strides):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = scale(self.rtm_reg(reg_feat).exp()).float() * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return tuple(cls_scores), tuple(bbox_preds)

    def loss_by_feat_single(  # type: ignore[override]
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        assign_metrics: Tensor,
        stride: list[int],
    ) -> tuple[Tensor, ...]:
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (list[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if stride[0] != stride[1]:
            msg = "h stride is not equal to w stride!"
            raise ValueError(msg)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0,
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.0)

        return loss_cls, loss_bbox, assign_metrics.sum(), pos_bbox_weight.sum()

    def loss_by_feat(  # type: ignore[override]
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        if len(featmap_sizes) != self.prior_generator.num_levels:
            msg = (
                f"The number of featmap_sizes (={len(featmap_sizes)}) and the number of levels of prior_generator "
                f"(={self.prior_generator.num_levels}) should be same."
            )
            raise ValueError(msg)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat(
            [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores],
            1,
        )
        decoded_bboxes = []
        for anchor, bbox_pred in zip(anchor_list[0], bbox_preds):
            anchor = anchor.reshape(-1, 4)  # noqa: PLW2901
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)  # noqa: PLW2901
            bbox_pred = distance2bbox(anchor, bbox_pred)  # noqa: PLW2901
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            sampling_results_list,
        ) = self.get_targets(  # type: ignore[misc]
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )

        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            decoded_bboxes,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            self.prior_generator.strides,
        )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = [x / cls_avg_factor for x in losses_cls]

        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = [x / bbox_avg_factor for x in losses_bbox]
        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox}

    def export_by_feat(  # type: ignore[override]
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Transform network output for a batch into bbox predictions.

        Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/dense_heads/rtmdet_head.py#L18-L108

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (dict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
                where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
                size and the score between 0 and 1. The shape of the second
                tensor in the tuple is (N, num_box), and each element
                represents the class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)  # noqa: S101
        device = cls_scores[0].device
        cfg = self.test_cfg if cfg is None else cfg
        batch_size = bbox_preds[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, device=device)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bbox_pred in bbox_preds]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        priors = torch.cat(mlvl_priors)
        tl_x = priors[..., 0] - flatten_bbox_preds[..., 0]  # type: ignore[call-overload]
        tl_y = priors[..., 1] - flatten_bbox_preds[..., 1]  # type: ignore[call-overload]
        br_x = priors[..., 0] + flatten_bbox_preds[..., 2]  # type: ignore[call-overload]
        br_y = priors[..., 1] + flatten_bbox_preds[..., 3]  # type: ignore[call-overload]
        bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        scores = flatten_cls_scores
        if not with_nms:
            return bboxes, scores

        return multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class=200,
            iou_threshold=cfg["nms"]["iou_threshold"],  # type: ignore[index]
            score_threshold=cfg["score_thr"],  # type: ignore[index]
            pre_top_k=5000,
            keep_top_k=cfg["max_per_img"],  # type: ignore[index]
        )

    def get_targets(  # type: ignore[override]
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | list[None] | None = None,
        unmap_outputs: bool = True,
    ) -> tuple | None:
        """Compute regression and classification targets for anchors in multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: a tuple containing learning targets.

            - anchors_list (list[list[Tensor]]): Anchors of each level.
            - labels_list (list[Tensor]): Labels of each level.
            - label_weights_list (list[Tensor]): Label weights of each
              level.
            - bbox_targets_list (list[Tensor]): BBox targets of each level.
            - assign_metrics_list (list[Tensor]): alignment metrics of each
              level.
        """
        num_imgs = len(batch_img_metas)
        if len(anchor_list) != len(valid_flag_list) != num_imgs:
            msg = (
                f"The number of anchor_list (={len(anchor_list)}), the number of valid_flag_list "
                f"(={len(valid_flag_list)}), and num_imgs (={num_imgs}) should be same."
            )
            raise ValueError(msg)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            if len(anchor_list[i]) != len(valid_flag_list[i]):
                msg = (
                    f"The number of anchor_list[i] (={len(anchor_list[i])}) and the number of valid_flag_list[i] "
                    f"(={len(valid_flag_list[i])}) should be same."
                )
                raise ValueError(msg)
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])
        (
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_assign_metrics,
            sampling_results_list,
        ) = multi_apply(
            self._get_targets_single,
            cls_scores.detach(),
            bbox_preds.detach(),
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs,
        )
        # no valid anchors
        if any(labels is None for labels in all_labels):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        assign_metrics_list = images_to_levels(all_assign_metrics, num_level_anchors)

        return (
            anchors_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            sampling_results_list,
        )

    def _get_targets_single(  # type: ignore[override]
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        flat_anchors: Tensor,
        valid_flags: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: InstanceData | None = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Compute regression, classification targets for anchors in a single image.

        Args:
            cls_scores (list[Tensor]): Box scores for each image.
            bbox_preds (list[Tensor]): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (InstanceData): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (InstanceData, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with shape
              (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 4).
            - norm_alignment_metrics (Tensor): Normalized alignment metrics
              of all priors in the image with shape (N,).
            - sampling_result (SamplingResult): Sampling result.
        """
        inside_flags = anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.train_cfg["allowed_border"],
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors,
        )

        assign_result = self.assigner.assign(pred_instances, gt_instances, gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg["pos_weight"] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg["pos_weight"]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds == gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, assign_metrics, sampling_result)

    def get_anchors(
        self,
        featmap_sizes: list[tuple[int, int]],
        batch_img_metas: list[dict],
        device: torch.device | str = "cuda",
    ) -> tuple[list[list[Tensor]], list[list[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device or str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

            - anchor_list (list[list[Tensor]]): Anchors of each image.
            - valid_flag_list (list[list[Tensor]]): Valid flags of each
              image.
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device, with_stride=True)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_meta in batch_img_metas:
            multi_level_flags = self.prior_generator.valid_flags(featmap_sizes, img_meta["img_shape"], device)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list


class RTMDetSepBNHead(RTMDetHead):
    """RTMDetHead with separated BN layers and shared conv layers.

    Args:
        num_classes (int): Number of categories excluding the background category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution in head.
            Defaults to False.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.SiLU``.
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
        exp_on_reg (bool): Whether using exponential of regression features or not. Defaults to False.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        share_conv: bool = True,
        use_depthwise: bool = False,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] = nn.SiLU,
        pred_kernel_size: int = 1,
        exp_on_reg: bool = False,
        **kwargs,
    ) -> None:
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        self.use_depthwise = use_depthwise
        super().__init__(
            num_classes,
            in_channels,
            normalization=normalization,
            activation=activation,
            pred_kernel_size=pred_kernel_size,
            **kwargs,
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        conv = DepthwiseSeparableConvModule if self.use_depthwise else Conv2dModule
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        if self.with_objectness:
            self.rtm_obj = nn.ModuleList()
        for _ in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    conv(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                        activation=build_activation_layer(self.activation),
                    ),
                )
                reg_convs.append(
                    conv(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                        activation=build_activation_layer(self.activation),
                    ),
                )
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                ),
            )
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                ),
            )
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(self.feat_channels, 1, self.pred_kernel_size, padding=self.pred_kernel_size // 2),
                )

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg in zip(self.rtm_cls, self.rtm_reg):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

    def forward(self, feats: tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction

            - cls_scores (tuple[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_anchors * num_classes.
            - bbox_preds (tuple[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for idx, (x, stride) in enumerate(zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]
            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return tuple(cls_scores), tuple(bbox_preds)
