# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""MaskDINO Instance Segmentation model.

Implementation modified from:
    * https://github.com/IDEA-Research/MaskDINO
    * https://github.com/facebookresearch/Mask2Former
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.roi_align import RoIAlign

from otx.algo.common.utils.assigners import HungarianMatcher
from otx.algo.instance_segmentation.backbones.detectron_resnet import build_resnet_backbone
from otx.algo.instance_segmentation.heads.maskdino_head import MaskDINOHead
from otx.algo.instance_segmentation.heads.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from otx.algo.instance_segmentation.heads.transformer_decoder.maskdino_decoder import MaskDINODecoder
from otx.algo.instance_segmentation.losses import MaskDINOCriterion
from otx.algo.instance_segmentation.utils import box_ops
from otx.algo.instance_segmentation.utils.utils import ShapeSpec
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity
from otx.core.utils.mask_util import polygon_to_bitmap


class MaskDINO(nn.Module):
    """Main class for mask classification semantic segmentation architectures."""

    def __init__(
        self,
        backbone: nn.Module,
        sem_seg_head: MaskDINOHead,
        criterion: nn.Module,
        num_queries: int = 300,
        test_topk_per_image: int = 100,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.test_topk_per_image = test_topk_per_image
        self.roi_align = RoIAlign(
            output_size=(28, 28),
            sampling_ratio=0,
            aligned=True,
            spatial_scale=1.0,
        )

    @classmethod
    def from_config(cls, num_classes: int) -> MaskDINO:
        """Build a MaskDINO model from a config."""
        # Loss parameters:
        no_object_weight = 0.1

        # loss weights
        class_weight = 4.0
        cost_class_weight = 4.0
        cost_dice_weight = 5.0
        dice_weight = 5.0
        cost_mask_weight = 5.0
        mask_weight = 5.0
        cost_box_weight = 5.0
        box_weight = 5.0
        cost_giou_weight = 2.0
        giou_weight = 2.0
        train_num_points = 112 * 112
        oversample_ratio = 3.0
        importance_sample_ratio = 0.75

        dec_layers = 9

        backbone = build_resnet_backbone(
            norm="FrozenBN",
            stem_out_channels=64,
            input_shape=ShapeSpec(channels=3),
            freeze_at=0,
            out_features=("res2", "res3", "res4", "res5"),
            depth=50,
            num_groups=1,
            width_per_group=64,
            in_channels=64,
            out_channels=256,
            stride_in_1x1=False,
            res5_dilation=1,
        )

        sem_seg_head = MaskDINOHead(
            ignore_value=255,
            num_classes=num_classes,
            pixel_decoder=MaskDINOEncoder(
                input_shape=backbone.output_shape(),
                conv_dim=256,
                mask_dim=256,
                norm="GN",
                transformer_dropout=0.0,
                transformer_nheads=8,
                transformer_dim_feedforward=2048,
                transformer_enc_layers=6,
                transformer_in_features=["res3", "res4", "res5"],
                common_stride=4,
                total_num_feature_levels=4,
                num_feature_levels=3,
            ),
            loss_weight=1.0,
            transformer_predictor=MaskDINODecoder(
                num_classes=num_classes,
                hidden_dim=256,
                num_queries=300,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=9,
                mask_dim=256,
                noise_scale=0.4,
                dn_num=100,
                total_num_feature_levels=4,
            ),
        )

        matcher = HungarianMatcher(
            cost_dict={
                "cost_class": cost_class_weight,
                "cost_bbox": cost_box_weight,
                "cost_giou": cost_giou_weight,
                "cost_mask": cost_mask_weight,
                "cost_dice": cost_dice_weight,
            },
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_dice": dice_weight,
            "loss_mask": mask_weight,
            "loss_bbox": box_weight,
            "loss_giou": giou_weight,
        }
        weight_dict.update({k + "_interm": v for k, v in weight_dict.items()})

        # denoising training
        weight_dict.update({k + "_dn": v for k, v in weight_dict.items()})

        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        # building criterion
        criterion = MaskDINOCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=["labels", "masks", "boxes"],
            num_points=train_num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            dn_losses=["labels", "masks", "boxes"],
        )

        return MaskDINO(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=300,
            test_topk_per_image=100,
        )

    def forward(
        self,
        entity: InstanceSegBatchDataEntity,
        mode: str = "tensor",
    ) -> dict[str, Tensor]:
        """Forward pass."""
        img_shapes = [img_info.img_shape for img_info in entity.imgs_info]
        images = ImageList(entity.images, img_shapes)

        features = self.backbone(images.tensors)

        if self.training:
            targets = []
            for img_info, bboxes, labels, polygons, gt_masks in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
                entity.polygons,
                entity.masks,
            ):
                masks = polygon_to_bitmap(polygons, *img_info.img_shape) if len(polygons) else gt_masks
                norm_shape = torch.tile(torch.tensor(img_info.img_shape, device=img_info.device), (2,))
                targets.append(
                    {
                        "boxes": box_ops.box_xyxy_to_cxcywh(bboxes) / norm_shape,
                        "labels": labels,
                        "masks": tv_tensors.Mask(masks, device=img_info.device, dtype=torch.bool),
                    },
                )

            outputs, mask_dict = self.sem_seg_head(features, targets=targets)
            losses = self.criterion(outputs, targets, mask_dict)
            for k in list(losses.keys()):
                losses[k] *= self.criterion.weight_dict[k]
            return losses

        outputs, _ = self.sem_seg_head(features)
        return outputs

    def roi_mask_extraction(
        self,
        bboxes: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Extract masks from RoI (Region of Interest).

        This function is used for exporting the model, as it extracts same-size square masks from RoI for speed.

        Args:
            bboxes (Tensor): Bounding boxes with shape (N, 4), where N is the number of bounding boxes.
            masks (Tensor): Masks with shape (H, W), where H and W are the height and width of the mask.

        Returns:
            Tensor: Extracted masks with shape (1, N, H', W'), where H' and W'
                are the height and width of the extracted masks.
        """
        bboxes = bboxes.unsqueeze(0)
        batch_index = torch.arange(bboxes.size(0)).float().view(-1, 1, 1).expand(bboxes.size(0), bboxes.size(1), 1)
        rois = torch.cat([batch_index, bboxes], dim=-1)
        cropped_masks = self.roi_align(masks.unsqueeze(0), rois[0])
        cropped_masks = cropped_masks[torch.arange(cropped_masks.size(0)), torch.arange(cropped_masks.size(0))]
        return (cropped_masks > 0).unsqueeze(0)

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Export the model."""
        b, _, h, w = batch_inputs.size()
        if b != 1:
            msg = "Only support batch size 1 for export"
            raise ValueError(msg)

        features = self.backbone(batch_inputs)
        outputs, _ = self.sem_seg_head(features)
        mask_cls = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]

        num_classes = self.sem_seg_head.num_classes
        num_queries = self.num_queries
        test_topk_per_image = self.test_topk_per_image

        scores = mask_cls.sigmoid()
        labels = torch.arange(num_classes).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes

        mask_pred = torch.nn.functional.interpolate(
            mask_pred.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]

        mask_pred = mask_pred[topk_indices]
        pred_boxes = pred_boxes[topk_indices]
        pred_masks = (mask_pred > 0).float()

        # Calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
            pred_masks.flatten(1).sum(1) + 1e-6
        )
        pred_scores = scores_per_image * mask_scores_per_image
        pred_classes = labels_per_image.unsqueeze(0)
        pred_boxes = pred_boxes.new_tensor([[w, h, w, h]]) * box_ops.box_cxcywh_to_xyxy(pred_boxes)
        pred_masks = self.roi_mask_extraction(pred_boxes, pred_masks)

        boxes_with_scores = torch.cat([pred_boxes, pred_scores[:, None]], dim=1)
        boxes_with_scores = boxes_with_scores.unsqueeze(0)

        return (
            boxes_with_scores,
            pred_classes,
            pred_masks,
        )
