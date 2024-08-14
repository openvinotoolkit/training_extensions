# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.dense_heads.yolox_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/dense_heads/yolox_head.py
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Callable, Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision.ops import box_convert

from otx.algo.common.losses import CrossEntropyLoss, L1Loss
from otx.algo.common.utils.nms import batched_nms, multiclass_nms
from otx.algo.common.utils.prior_generators import MlvlPointGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.common.utils.utils import multi_apply, reduce_mean
from otx.algo.detection.heads.base_head import BaseDenseHead
from otx.algo.detection.losses import IoULoss
from otx.algo.modules.activation import Swish, build_activation_layer
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.modules.norm import build_norm_layer
from otx.algo.utils.mmengine_utils import InstanceData

logger = logging.getLogger()


class YOLOXHead(BaseDenseHead):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Defaults to 256
        stacked_convs (int): Number of stacking convs of the head.
            Defaults to (8, 16, 32).
        strides (Sequence[int]): Downsample factor of each feature map.
            Defaults to None.
        use_depthwise (bool): Whether to depthwise separable convolution in blocks.
            Defaults to False.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of towers.
            Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the normalization. Bias of conv will be set as True if `normalization` is
            None, otherwise False. Defaults to "auto".
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``Swish``.
        loss_cls (nn.Module, optional): Module of classification loss.
        loss_bbox (nn.Module, optional): Module of localization loss.
        loss_obj (nn.Module, optional): Module of objectness loss.
        loss_l1 (nn.Module, optional): Module of L1 loss.
        train_cfg (dict, optional): Training config of anchor head.
            Defaults to None.
        test_cfg (dict, optional): Testing config of anchor head.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32),
        use_depthwise: bool = False,
        dcn_on_last_conv: bool = False,
        conv_bias: bool | str = "auto",
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] = Swish,
        loss_cls: nn.Module | None = None,
        loss_bbox: nn.Module | None = None,
        loss_obj: nn.Module | None = None,
        loss_l1: nn.Module | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        if init_cfg is None:
            init_cfg = {
                "type": "Kaiming",
                "layer": "Conv2d",
                "a": math.sqrt(5),
                "distribution": "uniform",
                "mode": "fan_in",
                "nonlinearity": "leaky_relu",
            }

        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        if conv_bias != "auto" and not isinstance(conv_bias, bool):
            msg = f"conv_bias (={conv_bias}) should be bool or str."
            raise ValueError(msg)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.normalization = normalization
        self.activation = activation

        self.loss_cls = loss_cls or CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0)
        self.loss_bbox = loss_bbox or IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0)
        self.loss_obj = loss_obj or CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0)

        self.use_l1 = False  # This flag will be modified by hooks.
        self.loss_l1 = loss_l1 or L1Loss(reduction="sum", loss_weight=1.0)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)  # type: ignore[arg-type]

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        if self.train_cfg is not None:
            self.assigner = self.train_cfg["assigner"]
            # YOLOX does not support sampling
            self.sampler = PseudoSampler()  # type: ignore[no-untyped-call]

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize heads for all level feature maps."""
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self) -> nn.Sequential:
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule if self.use_depthwise else Conv2dModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # TODO (sungchul): enable deformable convolution implemented in mmcv
            # conv_cfg = {"type": "DCNv2"} if self.dcn_on_last_conv and i == self.stacked_convs - 1 else self.conv_cfg
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                logger.warning(
                    f"stacked convs[{i}] : Deformable convolution is not supported in YOLOXHead, "
                    "use normal convolution instead.",
                )

            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    normalization=build_norm_layer(self.normalization, num_features=self.feat_channels),
                    activation=build_activation_layer(self.activation),
                    bias=self.conv_bias,
                ),
            )
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def forward_single(
        self,
        x: Tensor,
        cls_convs: nn.Module,
        reg_convs: nn.Module,
        conv_cls: nn.Module,
        conv_reg: nn.Module,
        conv_obj: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, objectness

    def forward(self, x: tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return multi_apply(
            self.forward_single,
            x,
            self.multi_level_cls_convs,
            self.multi_level_reg_convs,
            self.multi_level_conv_cls,
            self.multi_level_conv_reg,
            self.multi_level_conv_obj,
        )

    def predict_by_feat(  # type: ignore[override]
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        objectnesses: list[Tensor] | None,
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> list[InstanceData]:
        """Transform a batch of output features extracted by the head into bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
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
            list[InstanceData]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)  # type: ignore[arg-type] # noqa: S101
        cfg = cfg or self.test_cfg

        num_imgs = len(batch_img_metas)  # type: ignore[arg-type]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]  # type: ignore[union-attr]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):  # type: ignore[arg-type]
            max_scores, labels = torch.max(flatten_cls_scores[img_id], 1)
            valid_mask = flatten_objectness[img_id] * max_scores >= cfg["score_thr"]  # type: ignore[index]
            results = InstanceData(
                bboxes=flatten_bboxes[img_id][valid_mask],
                scores=max_scores[valid_mask] * flatten_objectness[img_id][valid_mask],
                labels=labels[valid_mask],
            )

            result_list.append(
                self._bbox_post_process(
                    results=results,
                    cfg=cfg,
                    rescale=rescale,
                    with_nms=with_nms,
                    img_meta=img_meta,
                ),
            )

        return result_list

    def export_by_feat(  # type: ignore[override]
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        objectnesses: list[Tensor],
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Transform network output for a batch into bbox predictions.

        Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/dense_heads/yolox_head.py#L18-L118

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
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
        device = cls_scores[0].device
        cfg = cfg or self.test_cfg
        batch_size = bbox_preds[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, device=device, with_stride=True)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bbox_pred in bbox_preds]
        flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(batch_size, -1) for objectness in objectnesses]

        cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        score_factor = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        # directly multiply score factor and feed to nms
        scores = cls_scores * (score_factor.unsqueeze(-1))

        if not with_nms:
            return bboxes, scores

        return multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class=200,  # TODO (sungchul): temporarily set to mmdeploy cfg, will be updated
            iou_threshold=cfg["nms"]["iou_threshold"],  # type: ignore[index]
            score_threshold=cfg["score_thr"],  # type: ignore[index]
            pre_top_k=5000,
            keep_top_k=cfg["max_per_img"],  # type: ignore[index]
        )

    def _bbox_decode(self, priors: Tensor, bbox_preds: Tensor) -> Tensor:
        """Decode regression results (delta_x, delta_x, w, h) to bboxes (tl_x, tl_y, br_x, br_y).

        Args:
            priors (Tensor): Center proiors of an image, has shape (num_instances, 2).
            bbox_preds (Tensor): Box energies / deltas for all instances, has shape (batch_size, num_instances, 4).

        Returns:
            Tensor: Decoded bboxes in (tl_x, tl_y, br_x, br_y) format. Has
            shape (batch_size, num_instances, 4).
        """
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2

        return torch.stack([tl_x, tl_y, br_x, br_y], -1)

    def _bbox_post_process(  # type: ignore[override]
        self,
        results: InstanceData,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: dict | None = None,
    ) -> InstanceData:
        """Bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (InstaceData): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (dict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            InstanceData: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get("scale_factor") is not None  # type: ignore[union-attr] # noqa: S101
            results.bboxes /= results.bboxes.new_tensor(img_meta["scale_factor"][::-1]).repeat((1, 2))  # type: ignore[attr-defined, index]

        if with_nms and results.bboxes.numel() > 0:  # type: ignore[attr-defined]
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores, results.labels, cfg["nms"])  # type: ignore[attr-defined, index]
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results

    def loss_by_feat(  # type: ignore[override]
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        objectnesses: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: Sequence[InstanceData] | None = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[InstanceData], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs  # type: ignore[list-item]

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )

        flatten_cls_preds = torch.cat(
            [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_pred in cls_scores],
            dim=1,
        )
        flatten_bbox_preds = torch.cat(
            [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds],
            dim=1,
        )
        flatten_objectness = torch.cat(
            [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses],
            dim=1,
        )
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, num_fg_imgs) = multi_apply(
            self._get_targets_single,
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_cls_preds.detach(),
            flatten_bboxes.detach(),
            flatten_objectness.detach(),
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        # The experimental results show that 'reduce_mean' can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        if num_pos > 0:
            loss_cls = (
                self.loss_cls(flatten_cls_preds.view(-1, self.num_classes)[pos_masks], cls_targets) / num_total_samples
            )
            loss_bbox = self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks], bbox_targets) / num_total_samples
        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0

        loss_dict = {"loss_cls": loss_cls, "loss_bbox": loss_bbox, "loss_obj": loss_obj}

        if self.use_l1:
            if num_pos > 0:
                loss_l1 = self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks], l1_targets) / num_total_samples
            else:
                # Avoid cls and reg branch not participating in the gradient
                # propagation when there is no ground-truth in the images.
                # For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_l1 = flatten_bbox_preds.sum() * 0
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @torch.no_grad()
    def _get_targets_single(
        self,
        priors: Tensor,
        cls_preds: Tensor,
        decoded_bboxes: Tensor,
        objectness: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: InstanceData | None = None,
    ) -> tuple:
        """Compute classification, regression, and objectness targets for priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            gt_instances (InstanceData): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (InstanceData, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            tuple:
                foreground_mask (list[Tensor]): Binary mask of foreground
                targets.
                cls_target (list[Tensor]): Classification targets of an image.
                obj_target (list[Tensor]): Objectness targets of an image.
                bbox_target (list[Tensor]): BBox targets of an image.
                l1_target (int): BBox L1 targets of an image.
                num_pos_per_img (int): Number of positive samples in an image.
        """
        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(bboxes=decoded_bboxes, scores=scores.sqrt_(), priors=offset_priors)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore,
        )

        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, num_pos_per_img)

    def _get_l1_target(self, l1_target: Tensor, gt_bboxes: Tensor, priors: Tensor, eps: float = 1e-8) -> Tensor:
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = box_convert(gt_bboxes, in_fmt="xyxy", out_fmt="cxcywh")
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
