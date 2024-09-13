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
        self.load_from: str = None
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

    def _create_model(self) -> DETR:
        # backbone
        backbone = BackboneBuilder(self.model_name)
        depthaware_transformer = DepthAwareTransformerBuilder(self.model_name)
        # depth prediction module
        depth_predictor = DepthPredictor(depth_num_bins=80, depth_min=1e-3, depth_max=60.0, hidden_dim=256)
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

        criterion = MonoDETRCriterion(
            num_classes=self.num_classes,
            focal_alpha=0.25,
            weight_dict=weight_dict)

        model = MonoDETR(
            backbone,
            depthaware_transformer,
            depth_predictor,
            num_classes=self.num_classes,
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
        mask = entity.mask_2d

        key_list = ['labels', 'bboxes_2d', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'bboxes_3d']
        for bz in range(len(entity.imgs_info)):
            target_dict = {}
            for key in key_list:
                val = getattr(entity, key)
                target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)

        breakpoint()
        img_sizes = np.array([img_info.img_shape for img_info in entity.imgs_info])

        return {
            "inputs": entity.images,
            "calibs": entity.calib_p2,
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

        breakpoint()
        return Det3DBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            bboxes_2d=outputs["boxes"],
            labels=outputs["labels"],
            calibs=outputs["calibs"],
            bboxes_3d=outputs["bboxes_3d"],
            size_2d=outputs["size_2d"],
            size_3d=outputs["size_3d"],
            src_size_3d=outputs["src_size_3d"],
            depth=outputs["depth"],
            heading_bin=outputs["heading_bin"],
            heading_res=outputs["heading_res"],
            mask_2d=outputs["mask_2d"],
            indices=outputs["indices"],
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
