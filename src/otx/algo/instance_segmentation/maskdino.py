# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Main class for OTX MaskDINO model."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from otx.algo.common.utils.assigners import HungarianMatcher
from otx.algo.instance_segmentation.backbones.detectron_resnet import build_resnet_backbone
from otx.algo.instance_segmentation.heads.maskdino_head import MaskDINOHead
from otx.algo.instance_segmentation.heads.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from otx.algo.instance_segmentation.heads.transformer_decoder.maskdino_decoder import MaskDINODecoder
from otx.algo.instance_segmentation.losses import MaskDINOCriterion
from otx.algo.instance_segmentation.segmentors import MaskDINO
from otx.algo.instance_segmentation.utils import box_ops
from otx.algo.instance_segmentation.utils.utils import ShapeSpec
from otx.algo.modules.norm import AVAILABLE_NORMALIZATION_LIST
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel

if TYPE_CHECKING:
    from torchvision import tv_tensors


class MaskDINOR50(ExplainableOTXInstanceSegModel):
    """OTX MaskDINO model with ResNet50 backbone."""

    load_from = "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth"
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def _build_model(self, num_classes: int) -> nn.Module:
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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 16,
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "masks"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for MaskDINO-R50."""
        return {"model_type": "transformer"}

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers."""
        param_groups = self._get_optim_params(self.model)
        optimizer = self.optimizer_callable(param_groups)

        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    def _get_optim_params(self, model: nn.Module) -> list[dict[str, Any]]:
        """Get optimizer parameters."""
        _optimizer = self.optimizer_callable(self.parameters())

        # Configurable from MaskDINO recipe
        base_lr = _optimizer.defaults["lr"]
        weight_decay = _optimizer.defaults["weight_decay"]
        defaults = {
            "lr": base_lr,
            "weight_decay": weight_decay,
        }

        # Static optimizer params
        weight_decay_norm = 0.0
        weight_decay_embed = 0.0
        backbone_multiplier = 0.1

        params = []
        uniques = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in uniques:
                    continue
                uniques.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * backbone_multiplier
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, AVAILABLE_NORMALIZATION_LIST):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
        return params

    def _customize_outputs(
        self,
        outputs: dict[str, Tensor],  # type: ignore[override]
        inputs: InstanceSegBatchDataEntity,
    ) -> OTXBatchLossEntity | InstanceSegBatchPredEntity:
        if self.training:
            return sum(outputs.values())

        masks, bboxes, labels, scores = self.post_process_instance_segmentation(
            outputs,
            inputs.imgs_info,
        )

        if self.explain_mode:
            msg = "Explain mode is not supported yet."
            raise NotImplementedError(msg)

        return InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    def post_process_instance_segmentation(
        self,
        outputs: dict[str, Tensor],
        imgs_info: list[ImageInfo],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Post-process MaskDINO outputs."""
        class_queries_logits = outputs["pred_logits"]
        masks_queries_logits = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]

        device = masks_queries_logits.device
        num_classes = self.model.sem_seg_head.num_classes
        num_queries = self.model.num_queries
        test_topk_per_image = self.model.test_topk_per_image

        batch_scores: list[Tensor] = []
        batch_bboxes: list[tv_tensors.BoundingBoxes] = []
        batch_labels: list[torch.LongTensor] = []
        batch_masks: list[tv_tensors.Mask] = []

        for mask_pred, mask_cls, pred_boxes, img_info in zip(
            masks_queries_logits,
            class_queries_logits,
            mask_box_results,
            imgs_info,
        ):
            ori_h, ori_w = img_info.ori_shape
            scores = mask_cls.sigmoid()
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes

            mask_pred = mask_pred[topk_indices]  # noqa: PLW2901
            pred_boxes = pred_boxes[topk_indices]  # noqa: PLW2901
            pred_scores = scores_per_image * self.calculate_mask_scores(mask_pred)
            pred_classes = labels_per_image

            pred_masks = (
                (
                    torch.nn.functional.interpolate(
                        mask_pred.unsqueeze(0),
                        size=(ori_h, ori_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                )
                > 0
            )

            pred_boxes = pred_boxes.new_tensor([[ori_w, ori_h, ori_w, ori_h]]) * box_ops.box_cxcywh_to_xyxy(pred_boxes)  # noqa: PLW2901
            pred_boxes[:, 0::2].clamp_(min=0, max=ori_w - 1)
            pred_boxes[:, 1::2].clamp_(min=0, max=ori_h - 1)

            area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
            keep = (pred_masks.sum((1, 2)) > 5) & (area > 10) & (pred_scores > 0.05)

            batch_masks.append(pred_masks[keep])
            batch_bboxes.append(pred_boxes[keep])
            batch_labels.append(pred_classes[keep])
            batch_scores.append(pred_scores[keep])

        return batch_masks, batch_bboxes, batch_labels, batch_scores

    def calculate_mask_scores(self, mask_pred: Tensor) -> Tensor:
        """Calculate mask scores."""
        pred_masks = (mask_pred > 0).to(mask_pred)

        # Calculate average mask prob
        return (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)
