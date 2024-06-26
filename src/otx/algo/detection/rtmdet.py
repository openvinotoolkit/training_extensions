# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDet model implementations."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torchvision import tv_tensors

from otx.algo.detection.backbones.cspnext import CSPNeXt
from otx.algo.detection.heads.base_sampler import PseudoSampler
from otx.algo.detection.heads.distance_point_bbox_coder import DistancePointBBoxCoder
from otx.algo.detection.heads.dynamic_soft_label_assigner import DynamicSoftLabelAssigner
from otx.algo.detection.heads.point_generator import MlvlPointGenerator
from otx.algo.detection.heads.rtmdet_head import RTMDetSepBNHead
from otx.algo.detection.losses.gfocal_loss import QualityFocalLoss
from otx.algo.detection.losses.iou_loss import GIoULoss
from otx.algo.detection.necks.cspnext_pafpn import CSPNeXtPAFPN
from otx.algo.detection.ssd import SingleStageDetector
from otx.algo.utils.mmengine_utils import InstanceData, load_checkpoint
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.types.export import TaskLevelExportParameters


class RTMDet(ExplainableOTXDetModel):
    """OTX Detection model class for RTMDet."""

    def _create_model(self) -> nn.Module:
        detector = self._build_model(num_classes=self.label_info.num_classes)
        detector.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _build_model(self, num_classes: int) -> nn.Module:
        raise NotImplementedError

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)
        inputs: dict[str, Any] = {}

        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"

        return inputs

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, torch.Tensor):
                    losses[k] = v
                else:
                    msg = f"Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores: list[torch.Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
            if not isinstance(prediction, InstanceData):
                raise TypeError(prediction)

            filtered_idx = torch.where(prediction.scores > self.best_confidence_threshold)  # type: ignore[attr-defined]
            scores.append(prediction.scores[filtered_idx])  # type: ignore[attr-defined]
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes[filtered_idx],  # type: ignore[attr-defined]
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
            labels.append(prediction.labels[filtered_idx])  # type: ignore[attr-defined]

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_map = outputs["saliency_map"].detach().cpu().numpy()
            feature_vector = outputs["feature_vector"].detach().cpu().numpy()

            return DetBatchPredEntity(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=saliency_map,
                feature_vector=feature_vector,
            )

        return DetBatchPredEntity(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def get_classification_layers(self, prefix: str = "") -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            prefix (str): Prefix of model param name.
                Normally it is "model." since OTXModel set it's nn.Module model as self.model

        Return:
            dict[str, dict[str, int]]
            A dictionary contain classification layer's name and information.
            Stride means dimension of each classes, normally stride is 1, but sometimes it can be 4
            if the layer is related bbox regression for object detection.
            Extra classes is default class except class from data.
            Normally it is related with background classes.
        """
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=self.mean,
            std=self.std,
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=True,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(optimization_config={"preset": "mixed"})

    def forward_for_tracing(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)


class RTMDetTiny(RTMDet):
    """RTMDet Tiny Model."""

    load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/rtmdet_tiny.pth"
    image_size = (1, 3, 640, 640)
    tile_image_size = (1, 3, 640, 640)
    mean = (103.53, 116.28, 123.675)
    std = (57.375, 57.12, 58.395)

    def _build_model(self, num_classes: int) -> RTMDet:
        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = DictConfig(
            {
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "score_thr": 0.001,
                "mask_thr_binary": 0.5,
                "max_per_img": 300,
                "min_bbox_size": 0,
                "nms_pre": 30000,
            },
        )

        backbone = CSPNeXt(
            arch="P5",
            expand_ratio=0.5,
            deepen_factor=0.167,
            widen_factor=0.375,
            channel_attention=True,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
        )

        neck = CSPNeXtPAFPN(
            in_channels=(96, 192, 384),
            out_channels=96,
            num_csp_blocks=1,
            expand_ratio=0.5,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
        )

        bbox_head = RTMDetSepBNHead(
            num_classes=num_classes,
            in_channels=96,
            stacked_convs=2,
            feat_channels=96,
            anchor_generator=MlvlPointGenerator(offset=0, strides=[8, 16, 32]),
            bbox_coder=DistancePointBBoxCoder(),
            loss_cls=QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0),
            loss_bbox=GIoULoss(loss_weight=2.0),
            with_objectness=False,
            exp_on_reg=False,
            share_conv=True,
            pred_kernel_size=1,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        return SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
