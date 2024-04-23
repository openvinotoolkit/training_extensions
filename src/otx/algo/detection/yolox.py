# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal
from torch import nn

from otx.algo.detection.ssd import SingleStageDetector
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.mmdeploy import MMdeployExporter
from otx.core.metrics.mean_ap import MeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import MMDetCompatibleModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.utils.build import modify_num_classes
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.config import convert_conf_to_mmconfig_dict
from otx.core.utils.utils import get_mean_std_from_data_processing
from otx.core.model.utils.mmdet import DetDataPreprocessor

from otx.algo.detection.backbones.csp_darknet import CSPDarknet
from otx.algo.detection.necks.yolox_pafpn import YOLOXPAFPN
from otx.algo.detection.heads.yolox_head import YOLOXHead

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmengine import ConfigDict

    from otx.core.metrics import MetricCallable
    from omegaconf import DictConfig


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
        return YOLOXHead(**cfg)

    def build_det_data_preprocessor(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build DetDataPreprocessor.

        TODO (sungchul): DetDataPreprocessor will be removed.
        """
        cfg.pop("type")  # TODO (sungchul): remove `type` in recipe
        return DetDataPreprocessor(**cfg)


class OTXYOLOX(MMDetCompatibleModel):
    """OTX Detection model class for YOLOX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        variant: Literal["tiny", "l", "s", "x"],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAPCallable,
        torch_compile: bool = False,
    ) -> None:
        self.variant = variant
        model_name = f"yolox_{self.variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(
            label_info=label_info,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
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

    @staticmethod
    def get_classification_layers(
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
