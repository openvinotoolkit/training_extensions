# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ATSS model implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

import torch
from mmengine.structures import InstanceData
from torchvision import tv_tensors

from otx.algo.detection.backbones.pytorchcv_backbones import _build_model_including_pytorchcv
from otx.algo.detection.backbones.resnext import ResNeXt
from otx.algo.detection.heads.atss_head import ATSSHead
from otx.algo.detection.necks.fpn import FPN
from otx.algo.detection.ssd import SingleStageDetector
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel, MMDetCompatibleModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.config import convert_conf_to_mmconfig_dict, inplace_num_classes
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmengine import ConfigDict
    from torch import Tensor, nn

    from otx.core.metrics import MetricCallable


class TorchATSS(SingleStageDetector):
    """ATSS torch implementation."""

    def __init__(self, neck: ConfigDict | dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.neck = self.build_neck(neck)

    def build_backbone(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build backbone."""
        if cfg["type"] == "ResNeXt":
            cfg.pop("type")
            return ResNeXt(**cfg)
        return _build_model_including_pytorchcv(cfg)

    def build_neck(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build backbone."""
        return FPN(**cfg)

    def build_bbox_head(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build bbox head."""
        return ATSSHead(**cfg)


class ATSS(MMDetCompatibleModel):
    """ATSS Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        variant: Literal["mobilenetv2", "resnext101"],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        model_name = f"atss_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(
            label_info=label_info,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )
        self.image_size = (1, 3, 736, 992)
        self.tile_image_size = self.image_size
        self._classification_layers: dict[str, dict[str, int]] | None = None

    def _create_model(self) -> nn.Module:
        from mmengine.runner import load_checkpoint

        config = deepcopy(self.config)
        self.classification_layers = self.get_classification_layers()
        model = TorchATSS(**convert_conf_to_mmconfig_dict(config))
        if self.load_from is not None:
            load_checkpoint(model, self.load_from, map_location="cpu")
        return model

    def get_classification_layers(self, prefix: str = "model.") -> dict[str, dict[str, int]]:
        """Get final classification layer information for incremental learning case."""
        from otx.core.utils.build import modify_num_classes

        sample_config = deepcopy(self.config)
        modify_num_classes(sample_config, 5)
        sample_model_dict = TorchATSS(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()
        modify_num_classes(sample_config, 6)
        incremental_model_dict = TorchATSS(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()

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

        mean, std = get_mean_std_from_data_processing(self.config)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # Currently ATSS should be exported through ONNX
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
            output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> list[InstanceData]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        sample = InstanceData(
            metainfo=meta_info,
        )
        data_samples = [sample] * len(inputs)
        return self.model.export(inputs, data_samples)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)


class OTXATSS(ExplainableOTXDetModel):
    """ATSS Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        variant: Literal["mobilenetv2", "resnext101"],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        model_name = f"atss_{variant}"
        config = read_mmconfig(model_name=model_name)
        config = inplace_num_classes(cfg=config, num_classes=self._dispatch_label_info(label_info).num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )
        self.image_size = (1, 3, 736, 992)
        self.tile_image_size = self.image_size

    def _create_model(self) -> nn.Module:
        from mmengine.runner import load_checkpoint

        config = deepcopy(self.config)
        self.classification_layers = self.get_classification_layers()
        model = TorchATSS(**convert_conf_to_mmconfig_dict(config))
        if self.load_from is not None:
            load_checkpoint(model, self.load_from, map_location="cpu")
        return model

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        from mmdet.structures import DetDataSample

        mmdet_inputs: dict[str, Any] = {}

        mmdet_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmdet_inputs["data_samples"] = [
            DetDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                    "ignored_labels": img_info.ignored_labels,
                },
                gt_instances=InstanceData(
                    bboxes=bboxes,
                    labels=labels,
                ),
            )
            for img_info, bboxes, labels in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
            )
        ]

        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs

    def _customize_outputs(
        self,
        outputs: dict[str, Any],
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        from mmdet.structures import DetDataSample

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
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores = []
        bboxes = []
        labels = []

        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for output in predictions:
            if not isinstance(output, DetDataSample):
                raise TypeError(output)
            scores.append(output.pred_instances.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    output.pred_instances.bboxes,
                    format="XYXY",
                    canvas_size=output.ori_shape,
                ),
            )
            labels.append(output.pred_instances.labels)

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

    def get_classification_layers(self, prefix: str = "model.") -> dict[str, dict[str, int]]:
        """Get final classification layer information for incremental learning case."""
        from otx.core.utils.build import modify_num_classes

        sample_config = deepcopy(self.config)
        modify_num_classes(sample_config, 5)
        sample_model_dict = TorchATSS(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()
        modify_num_classes(sample_config, 6)
        incremental_model_dict = TorchATSS(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()

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

        mean, std = get_mean_std_from_data_processing(self.config)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # Currently ATSS should be exported through ONNX
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
            output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> list[InstanceData]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        sample = InstanceData(
            metainfo=meta_info,
        )
        data_samples = [sample] * len(inputs)
        return self.model.export(inputs, data_samples)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)
