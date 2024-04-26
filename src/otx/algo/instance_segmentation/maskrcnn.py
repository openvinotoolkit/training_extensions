# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskRCNN model implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from mmengine.structures import InstanceData

from otx.algo.instance_segmentation.mmdet.models.detectors import MaskRCNN
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.mmdeploy import MMdeployExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import MMDetInstanceSegCompatibleModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.build import modify_num_classes
from otx.core.utils.config import convert_conf_to_mmconfig_dict
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from omegaconf import DictConfig
    from torch.nn.modules import Module

    from otx.core.metrics import MetricCallable


class MMDetMaskRCNN(MMDetInstanceSegCompatibleModel):
    """MaskRCNN Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        variant: Literal["efficientnetb2b", "r50"],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        model_name = f"maskrcnn_{variant}"
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
        self.image_size = (1, 3, 1024, 1024)
        self.tile_image_size = (1, 3, 512, 512)

    def get_classification_layers(self, config: DictConfig, prefix: str = "") -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            config (DictConfig): Config for building model.
            model_registry (Registry): Registry for building model.
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
        modify_num_classes(sample_config, 5)
        sample_model_dict = MaskRCNN(
            **convert_conf_to_mmconfig_dict(sample_config, to="list"),
        ).state_dict()

        modify_num_classes(sample_config, 6)
        incremental_model_dict = MaskRCNN(
            **convert_conf_to_mmconfig_dict(sample_config, to="list"),
        ).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

    def _create_model(self) -> Module:
        from mmengine.runner import load_checkpoint

        config = deepcopy(self.config)
        self.classification_layers = self.get_classification_layers(config, "model.")
        detector = MaskRCNN(**convert_conf_to_mmconfig_dict(config, to="list"))
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        mean, std = get_mean_std_from_data_processing(self.config)

        # with self.export_model_forward_context():
        #     return MMdeployExporter(
        #         model_builder=self._create_model,
        #         model_cfg=deepcopy(self.config),
        #         deploy_cfg="otx.algo.instance_segmentation.mmdeploy.maskrcnn",
        #         test_pipeline=self._make_fake_test_pipeline(),
        #         task_level_export_parameters=self._export_parameters,
        #         input_size=self.image_size,
        #         mean=mean,
        #         std=std,
        #         resize_mode="standard",  # [TODO](@Eunwoo): need to revert it to fit_to_window after resolving
        #         pad_value=0,
        #         swap_rgb=False,
        #         output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
        #     )

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 11,
                "autograd_inlining": False,
            },
            output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(
        self,
        inputs: torch.Tensor,
    ) -> list[InstanceData]:
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
        return self.model.export(
            inputs,
            data_samples,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_iseg_ckpt(state_dict, add_prefix)


class MaskRCNNSwinT(MMDetInstanceSegCompatibleModel):
    """MaskRCNNSwinT Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        model_name = "maskrcnn_swint"
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
        self.image_size = (1, 3, 1344, 1344)
        self.tile_image_size = (1, 3, 512, 512)

    def get_classification_layers(self, config: DictConfig, prefix: str = "") -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            config (DictConfig): Config for building model.
            model_registry (Registry): Registry for building model.
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
        modify_num_classes(sample_config, 5)
        sample_model_dict = MaskRCNN(**convert_conf_to_mmconfig_dict(sample_config, to="list")).state_dict()

        modify_num_classes(sample_config, 6)
        incremental_model_dict = MaskRCNN(
            **convert_conf_to_mmconfig_dict(sample_config, to="list"),
        ).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

    def _create_model(self) -> Module:
        from mmengine.runner import load_checkpoint

        config = deepcopy(self.config)
        self.classification_layers = self.get_classification_layers(config, "model.")
        detector = MaskRCNN(**convert_conf_to_mmconfig_dict(config, to="list"))
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

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
                deploy_cfg="otx.algo.instance_segmentation.mmdeploy.maskrcnn_swint",
                test_pipeline=self._make_fake_test_pipeline(),
                task_level_export_parameters=self._export_parameters,
                input_size=self.image_size,
                mean=mean,
                std=std,
                resize_mode="standard",  # [TODO](@Eunwoo): need to revert it to fit_to_window after resolving
                pad_value=0,
                swap_rgb=False,
                output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
            )
