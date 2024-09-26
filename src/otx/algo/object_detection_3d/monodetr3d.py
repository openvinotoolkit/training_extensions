# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MonoDetr model implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert

from otx.algo.object_detection_3d.backbones.monodetr_resnet import BackboneBuilder
from otx.algo.object_detection_3d.detectors.monodetr import MonoDETR
from otx.algo.object_detection_3d.heads.depth_predictor import DepthPredictor
from otx.algo.object_detection_3d.heads.depthaware_transformer import DepthAwareTransformerBuilder
from otx.algo.object_detection_3d.losses import MonoDETRCriterion
from otx.algo.object_detection_3d.utils.utils import box_cxcylrtb_to_xyxy
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.detection_3d import OTXObjectDetection3DExporter
from otx.core.model.detection_3d import OTX3DDetectionModel


class MonoDETR3D(OTX3DDetectionModel):
    """OTX Detection model class for MonoDETR3D."""

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    input_size: tuple[int, int] = (384, 1280)  # HxW
    load_from: str | None = None

    def _build_model(self, num_classes: int) -> MonoDETR:
        # backbone
        backbone = BackboneBuilder(self.model_name)
        # transformer
        depthaware_transformer = DepthAwareTransformerBuilder(self.model_name)
        # depth prediction module
        depth_predictor = DepthPredictor(depth_num_bins=80, depth_min=1e-3, depth_max=60.0, hidden_dim=256)
        # criterion
        loss_weight_dict = {
            "loss_ce": 2,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_center": 10,
        }
        criterion = MonoDETRCriterion(num_classes=num_classes, focal_alpha=0.25, weight_dict=loss_weight_dict)

        return MonoDETR(
            backbone,
            depthaware_transformer,
            depth_predictor,
            num_classes=num_classes,
            criterion=criterion,
            num_queries=50,
            aux_loss=True,
            num_feature_levels=4,
            with_box_refine=True,
            init_box=False,
        )

    def _customize_inputs(
        self,
        entity: Det3DBatchDataEntity,
    ) -> dict[str, Any]:
        # prepare bboxes for the model
        targets_list = []
        img_sizes = torch.from_numpy(np.array([img_info.ori_shape for img_info in entity.imgs_info])).to(
            device=entity.images.device,
        )
        key_list = ["labels", "boxes", "depth", "size_3d", "heading_angle", "boxes_3d"]
        for bz in range(len(entity.imgs_info)):
            target_dict = {}
            for key in key_list:
                target_dict[key] = getattr(entity, key)[bz]
            targets_list.append(target_dict)

        return {
            "images": entity.images,
            "calibs": torch.cat([p2.unsqueeze(0) for p2 in entity.calib_matrix], dim=0),
            "targets": targets_list,
            "img_sizes": img_sizes,
            "mode": "loss" if self.training else "predict",
        }

    def _customize_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        inputs: Det3DBatchDataEntity,
    ) -> Det3DBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        labels, scores, size_3d, heading_angle, boxes_3d, depth = self.extract_dets_from_outputs(outputs)
        # bbox 2d decoding
        boxes_2d = box_cxcylrtb_to_xyxy(boxes_3d)
        xywh_2d = box_convert(boxes_2d, "xyxy", "cxcywh")
        # size 2d decoding
        size_2d = xywh_2d[:, :, 2:4]

        return Det3DBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            calib_matrix=inputs.calib_matrix,
            boxes=boxes_2d,
            labels=labels,
            boxes_3d=boxes_3d,
            size_2d=size_2d,
            size_3d=size_3d,
            depth=depth,
            heading_angle=heading_angle,
            scores=scores,
            kitti_label_object=inputs.kitti_label_object,
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        param_groups = self._apply_no_bias_decay()
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

    def _apply_no_bias_decay(self) -> list[dict[str, Any]]:
        """Apply no bias decay to bias parameters."""
        weights, biases = [], []
        for name, param in self.named_parameters():
            if "bias" in name:
                biases += [param]
            else:
                weights += [param]

        return [{"params": biases, "weight_decay": 0}, {"params": weights, "weight_decay": 0.0001}]

    def forward_for_tracing(
        self,
        images: torch.Tensor,
        calib_matrix: torch.Tensor,
        img_sizes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        outputs = self.model(images=images, calibs=calib_matrix, img_sizes=img_sizes, mode="predict")

        return {
            "scores": outputs["pred_logits"].sigmoid(),
            "boxes_3d": outputs["pred_boxes"],
            "size_3d": outputs["pred_3d_dim"],
            "heading_angle": outputs["pred_angle"],
            "depth": outputs["pred_depth"],  # depth + deviation
        }

    @staticmethod
    def extract_dets_from_outputs(outputs: dict[str, torch.Tensor], topk: int = 50) -> tuple[torch.Tensor, ...]:
        """Extract detection results from model outputs."""
        # b, q, c
        out_logits = outputs["pred_logits"]
        out_bbox = outputs["pred_boxes"]

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

        # final scores
        scores = topk_values
        # final indexes
        topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
        # final labels
        labels = topk_indexes % out_logits.shape[2]

        heading = outputs["pred_angle"]
        size_3d = outputs["pred_3d_dim"]
        depth = outputs["pred_depth"]
        # decode boxes
        boxes_3d = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4
        # heading angle decoding
        heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
        # depth decoding
        depth = torch.gather(depth, 1, topk_boxes.repeat(1, 1, 2))
        # 3d dims decoding
        size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))
        # 2d boxes of the corners decoding

        return labels, scores, size_3d, heading, boxes_3d, depth

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        return OTXObjectDetection3DExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images", "calib_matrix", "img_sizes"],
                "output_names": ["scores", "size_3d", "heading_angle", "boxes_3d", "depth"],
                "dynamic_axes": {
                    "images": {0: "batch"},
                    "boxes_3d": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                    "heading_angle": {0: "batch", 1: "num_dets"},
                    "depth": {0: "batch", 1: "num_dets"},
                    "size_3d": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
                "opset_version": 16,
            },
            input_names=["images", "calib_matrix", "img_sizes"],
            output_names=["scores", "boxes_3d", "size_3d", "heading_angle", "depth"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for MonoDETR."""
        return {"model_type": "transformer"}
