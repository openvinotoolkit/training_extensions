# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

import torch
from mmengine.structures import InstanceData
from torchvision import tv_tensors

from otx.algo.detection.backbones.csp_darknet import CSPDarknet
from otx.algo.detection.heads.yolox_head import YOLOXHead
from otx.algo.detection.necks.yolox_pafpn import YOLOXPAFPN
from otx.algo.detection.ssd import SingleStageDetector
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.mmdeploy import MMdeployExporter
from otx.core.metrics.mean_ap import MeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.model.utils.mmdet import DetDataPreprocessor
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.build import modify_num_classes
from otx.core.utils.config import convert_conf_to_mmconfig_dict, inplace_num_classes
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmengine import ConfigDict
    from omegaconf import DictConfig
    from torch import nn

    from otx.core.metrics import MetricCallable


class YOLOX(SingleStageDetector):
    """YOLOX implementation from mmdet."""

    def __init__(self, neck: ConfigDict | dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.neck = self.build_neck(neck)

    def build_backbone(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build backbone."""
        cfg.pop("type")  # TODO (sungchul): remove `type` in recipe
        return CSPDarknet(**cfg)

    def build_neck(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build backbone."""
        cfg.pop("type")  # TODO (sungchul): remove `type` in recipe
        return YOLOXPAFPN(**cfg)

    def build_bbox_head(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build bbox head."""
        cfg.pop("type")  # TODO (sungchul): remove `type` in recipe
        return YOLOXHead(**cfg)

    def build_det_data_preprocessor(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build DetDataPreprocessor.

        TODO (sungchul): DetDataPreprocessor will be removed.
        """
        cfg.pop("type")  # TODO (sungchul): remove `type` in recipe
        return DetDataPreprocessor(**cfg)


class OTXYOLOX(ExplainableOTXDetModel):
    """OTX Detection model class for YOLOX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        variant: Literal["tiny", "l", "s", "x"],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.variant = variant
        model_name = f"yolox_{self.variant}"
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
        self.image_size = (1, 3, 416, 416) if self.variant == "tiny" else (1, 3, 640, 640)
        self.tile_image_size = self.image_size

    def _create_model(self) -> nn.Module:
        from mmengine.runner import load_checkpoint

        config = deepcopy(self.config)
        self.classification_layers = self.get_classification_layers(config, "model.")
        config.pop("type")  # TODO (sungchul): remove `type` in recipe
        detector = YOLOX(**convert_conf_to_mmconfig_dict(config))
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        mmdet_inputs: dict[str, Any] = {}

        mmdet_inputs["entity"] = entity
        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs

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
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores = []
        bboxes = []
        labels = []
        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
            if not isinstance(prediction, InstanceData):
                raise TypeError(prediction)
            scores.append(prediction.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
            labels.append(prediction.labels)

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
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=saliency_map,
                feature_vector=feature_vector,
            )

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def get_classification_layers(
        self,
        config: DictConfig,
        prefix: str = "",
    ) -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        TODO (sungchul): it can be merged to otx.core.utils.build.get_classification_layers.

        Args:
            config (DictConfig): Config for building model.
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
        sample_config = deepcopy(config)
        sample_config.pop("type")  # TODO (sungchul): remove `type` in recipe
        modify_num_classes(sample_config, 5)
        sample_model_dict = YOLOX(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()

        modify_num_classes(sample_config, 6)
        incremental_model_dict = YOLOX(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()

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

        deploy_cfg = "otx.algo.detection.mmdeploy.yolox"
        swap_rgb = True
        if self.variant == "tiny":
            deploy_cfg += "_tiny"
            swap_rgb = False

        with self.export_model_forward_context():
            return MMdeployExporter(
                model_builder=self._create_model,
                model_cfg=deepcopy(self.config),
                deploy_cfg=deploy_cfg,
                test_pipeline=self._make_fake_test_pipeline(),
                task_level_export_parameters=self._export_parameters,
                input_size=self.image_size,
                mean=mean,
                std=std,
                resize_mode="fit_to_window_letterbox",
                pad_value=114,
                swap_rgb=swap_rgb,
                output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
            )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)

    # TODO(Sungchul): Remove below functions after changing exporter
    def _make_fake_test_pipeline(self) -> list[dict[str, Any]]:
        return [
            {"type": "LoadImageFromFile"},
            {"type": "Resize", "scale": [self.image_size[3], self.image_size[2]], "keep_ratio": True},  # type: ignore[index]
            {"type": "LoadAnnotations", "with_bbox": True},
            {
                "type": "PackDetInputs",
                "meta_keys": ["ori_filenamescale_factor", "ori_shape", "filename", "img_shape", "pad_shape"],
            },
        ]
