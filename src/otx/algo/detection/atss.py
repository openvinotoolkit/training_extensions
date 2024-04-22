# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ATSS model implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from otx.algo.detection.backbones.fpn import FPN
from otx.algo.detection.backbones.pytorchcv_backbones import _build_pytorchcv_model
from otx.algo.detection.heads.custom_atss_head import ATSSHead
from otx.algo.detection.ssd import SingleStageDetector
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.mmdeploy import MMdeployExporter
from otx.core.metrics.mean_ap import MeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import MMDetCompatibleModel
from otx.core.model.utils.mmdet import DetDataPreprocessor
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.config import convert_conf_to_mmconfig_dict
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmengine import ConfigDict
    from torch import nn

    from otx.core.metrics import MetricCallable


class TorchATSS(SingleStageDetector):
    """ATSS torch implementation."""

    def __init__(
        self,
        backbone: ConfigDict | dict,
        neck: ConfigDict | dict,
        bbox_head: ConfigDict | dict,
        data_preprocessor: ConfigDict | dict,
        train_cfg: ConfigDict | dict | None = None,
        test_cfg: ConfigDict | dict | None = None,
        init_cfg: ConfigDict | list[ConfigDict] | dict | list[dict] = None,
    ) -> None:
        super(SingleStageDetector, self).__init__()
        self._is_init = False
        self.backbone = _build_pytorchcv_model(**backbone)
        neck.pop("type")
        self.neck = FPN(**neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        bbox_head.pop("type")
        self.bbox_head = ATSSHead(**bbox_head)
        data_preprocessor.pop("type")
        self.data_preprocessor = DetDataPreprocessor(**data_preprocessor)
        self.init_cfg = init_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


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
        config = deepcopy(self.config)
        config.pop("type")
        model = TorchATSS(**convert_conf_to_mmconfig_dict(config))
        self.classification_layers = self.get_classification_layers()
        return model

    def get_classification_layers(self, prefix: str = "model.") -> dict[str, dict[str, int]]:
        """Get final classification layer information for incremental learning case."""
        from otx.core.utils.build import modify_num_classes

        sample_config = deepcopy(self.config)
        sample_config.pop("type")
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

        with self.export_model_forward_context():
            return MMdeployExporter(
                model_builder=self._create_model,
                model_cfg=deepcopy(self.config),
                deploy_cfg="otx.algo.detection.mmdeploy.atss",
                test_pipeline=self._make_fake_test_pipeline(),
                task_level_export_parameters=self._export_parameters,
                input_size=self.image_size,
                mean=mean,
                std=std,
                resize_mode="standard",
                pad_value=0,
                swap_rgb=False,
                output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
            )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)
