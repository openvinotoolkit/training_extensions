# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTDetr model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
import numpy as np

import torch
from torch import Tensor
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat

from otx.algo.detection.detectors import DETR
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.ap_3d import KittiMetric
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection_3d import OTX3DDetectionModel
from .backbone import BackboneBuilder
from otx.algo.object_detection_3d.losses import MonoDETRCriterion
from otx.algo.object_detection_3d.depthaware_transformer import DepthAwareTransformerBuilder
from otx.algo.object_detection_3d.depth_predictor import DepthPredictor
from otx.algo.object_detection_3d.monodetr import MonoDETR
from otx.algo.object_detection_3d.utils.box_ops import box_cxcylrtb_to_xyxy
from otx.core.data.dataset.kitti_utils import class2angle

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class MonoDETR3D(OTX3DDetectionModel):
    """OTX Detection model class for MonoDETR3D.
    """

    mean: tuple[float, float, float] = [0.485, 0.456, 0.406]
    std: tuple[float, float, float] = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name: Literal["monodetr_50"],
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (640, 640),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = KittiMetric,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.load_from: str = "/home/kprokofi/MonoDETR/checkpoint_best_2.pth"
        super().__init__(
            model_name=model_name,
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _build_model(self, num_classes) -> DETR:
        # backbone
        backbone = BackboneBuilder(self.model_name)
        depthaware_transformer = DepthAwareTransformerBuilder(self.model_name)
        # depth prediction module
        cfg_model={'num_classes': 3, 'return_intermediate_dec': True, 'device': 'cuda', 'backbone': 'resnet50', 'train_backbone': True,
        'num_feature_levels': 4, 'dilation': False, 'position_embedding': 'sine', 'masks': False, 'mode': 'LID', 'num_depth_bins': 80,
        'depth_min': '1e-3', 'depth_max': 60.0, 'with_box_refine': True, 'two_stage': False, 'use_dab': False, 'use_dn': False, 'two_stage_dino': False,
        'init_box': False, 'enc_layers': 3, 'dec_layers': 3, 'hidden_dim': 256, 'dim_feedforward': 256, 'dropout': 0.1, 'nheads': 8, 'num_queries': 50,
        'enc_n_points': 4, 'dec_n_points': 4, 'scalar': 5, 'label_noise_scale': 0.2, 'box_noise_scale': 0.4, 'num_patterns': 0, 'aux_loss': True,
        'cls_loss_coef': 2, 'focal_alpha': 0.25, 'bbox_loss_coef': 5, 'giou_loss_coef': 2, '3dcenter_loss_coef': 10, 'dim_loss_coef': 1, 'angle_loss_coef': 1,
        'depth_loss_coef': 1, 'depth_map_loss_coef': 1, 'set_cost_class': 2, 'set_cost_bbox': 5, 'set_cost_giou': 2, 'set_cost_3dcenter': 10}
        depth_predictor = DepthPredictor(cfg_model)
        weight_dict = {'loss_ce': 2,
                       'loss_bbox': 5,
                       'loss_giou': 2,
                       'loss_dim': 1,
                       'loss_angle': 1,
                       'loss_depth': 1,
                       'loss_center': 10,
                       'loss_depth_map': 1}

        aux_weight_dict = {}
        for i in range(depthaware_transformer.decoder.num_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        print("NUM_CLASSES: ", num_classes)
        criterion = MonoDETRCriterion(
            num_classes=num_classes,
            focal_alpha=0.25,
            weight_dict=weight_dict)

        model = MonoDETR(
            backbone,
            depthaware_transformer,
            depth_predictor,
            num_classes=num_classes,
            criterion=criterion,
            num_queries=50,
            aux_loss=True,
            num_feature_levels=4,
            with_box_refine=True,
            two_stage=False,
            init_box=False,
            use_dab=False,
            two_stage_dino=False)

        return model

    def _customize_inputs(
        self,
        entity: Det3DBatchDataEntity,
    ) -> dict[str, Any]:
        # prepare bboxes for the model
        targets_list = []
        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(len(entity.imgs_info)):
            target_dict = {}
            for key in key_list:
                target_dict[key] = getattr(entity, key)[bz]
            targets_list.append(target_dict)
        img_sizes = torch.tensor([img_info.ori_shape for img_info in entity.imgs_info]).to(device=entity.images.device)

        return {
            "images": entity.images,
            "calibs": torch.cat([torch.as_tensor(cal.P2, device=entity.images.device).unsqueeze(0) for cal in entity.calib]),
            "targets" : targets_list,
            "img_sizes": img_sizes,
            "mode": "loss" if self.training else "predict",
        }

    def _customize_outputs(
        self,
        outputs: list[torch.Tensor] | dict,
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

        labels, scores, size_3d, size_2d, heading_angle, boxes_2d, boxes_3d, output_depth = self.extract_dets_from_outputs(outputs, score_threshold=0)
        assert not torch.any(torch.isnan(boxes_3d))
        assert not torch.any(torch.isnan(boxes_2d))
        assert not torch.any(torch.isnan(heading_angle))
        assert not torch.any(torch.isnan(output_depth))
        assert not torch.any(torch.isnan(size_2d))
        assert not torch.any(torch.isnan(size_3d))
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isnan(labels))

        return Det3DBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            boxes=boxes_2d,
            calib=inputs.calib,
            calibs=inputs.calibs,
            labels=labels,
            boxes_3d=boxes_3d,
            size_2d=size_2d,
            size_3d=size_3d,
            depth=output_depth,
            heading_bin=inputs.heading_bin,
            heading_res=heading_angle,
            scores=scores,
            kitti_label_object=inputs.kitti_label_object,
        )

    @staticmethod
    def extract_dets_from_outputs(outputs, topk=50, score_threshold=0.2):
        # get src outputs

        # b, q, c
        out_logits = outputs['pred_logits']
        out_bbox = outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

        # final scores
        scores = topk_values
        # final indexes
        topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
        # final labels
        labels = topk_indexes % out_logits.shape[2]

        heading = outputs['pred_angle']
        size_3d = outputs['pred_3d_dim']
        depth = outputs['pred_depth'][:, :, 0: 1]
        sigma = outputs['pred_depth'][:, :, 1: 2]
        sigma = torch.exp(-sigma)


        # decode boxes
        boxes_3d = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4
        # heading angle decoding
        heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
        # depth decoding
        depth = torch.gather(depth, 1, topk_boxes)
        sigma = torch.gather(sigma, 1, topk_boxes)
        output_depth = torch.cat([depth, sigma], dim=2) # predicted averaged depth and its deviation
        # 3d dims decoding
        size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))
        # 2d boxes of the corners decoding
        boxes_2d = box_cxcylrtb_to_xyxy(boxes_3d)
        # size 2d decoding
        xywh_2d = box_convert(boxes_2d, "xyxy", "cxcywh")
        size_2d = xywh_2d[:, :, 2: 4]

        return labels, scores, size_3d, size_2d, heading, boxes_2d, boxes_3d, output_depth

        # xs3d = boxes[:, :, 0: 1]
        # ys3d = boxes[:, :, 1: 2]
        # xs2d = xywh_2d[:, :, 0: 1]
        # ys2d = xywh_2d[:, :, 1: 2]

        # batch = out_logits.shape[0]
        # labels = labels.view(batch, -1, 1)
        # scores = scores.view(batch, -1, 1)
        # xs2d = xs2d.view(batch, -1, 1)
        # ys2d = ys2d.view(batch, -1, 1)
        # xs3d = xs3d.view(batch, -1, 1)
        # ys3d = ys3d.view(batch, -1, 1)

        # detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)

        # return detections

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
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {
                    "images": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
                "opset_version": 16,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RT-DETR."""
        return {"model_type": "transformer"}
