# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ATSS model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from torch.export import Dim

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, GIoULoss
from getitune.backend.lightning.models.common.utils.coders import DeltaXYWHBBoxCoder
from getitune.backend.lightning.models.common.utils.prior_generators import AnchorGenerator
from getitune.backend.lightning.models.common.utils.samplers import PseudoSampler
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors import SingleStageDetector
from getitune.backend.lightning.models.detection.heads import ATSSHead
from getitune.backend.lightning.models.detection.losses import ATSSCriterion
from getitune.backend.lightning.models.detection.necks import FPN
from getitune.backend.lightning.models.detection.utils.assigners import ATSSAssigner
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class ATSS(LightningDetectionModel):
    """getitune Detection model class for ATSS.

    Attributes:
        pretrained_urls (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
            If None, uses _default_preprocessing_params.
        model_name (Literal, optional): Name of the model to use. Defaults to "atss_mobilenetv2".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MeanAveragePrecisionFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    pretrained_urls: ClassVar[dict[str, str]] = {
        "atss_mobilenetv2": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/"
        "object_detection/v2/mobilenet_v2-atss.pth",
        "atss_resnext101": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/"
        "object_detection/v2/resnext101_atss_070623.pth",
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "atss_mobilenetv2",
            "atss_resnext101",
        ] = "atss_mobilenetv2",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        if pretrained and model_name not in self.pretrained_urls:
            msg = f"Unsupported model: {model_name}. Supported models: {list(self.pretrained_urls.keys())}"
            raise ValueError(msg)

        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            export_nms=export_nms,
        )

    def _create_model(self, num_classes: int | None = None) -> SingleStageDetector:
        num_classes = num_classes if num_classes is not None else self.num_classes
        # initialize backbones
        train_cfg = {
            "assigner": ATSSAssigner(topk=9),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.6},
            "min_bbox_size": 0,
            "score_thr": 0.05,
            "max_per_img": 100,
            "nms_pre": 1000,
        }
        backbone = self._build_backbone(model_name=self.model_name)
        neck = FPN(model_name=self.model_name)
        bbox_head = ATSSHead(
            model_name=self.model_name,
            num_classes=num_classes,
            anchor_generator=AnchorGenerator(
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            train_cfg=train_cfg,  # TODO (Kirill): remove
            test_cfg=test_cfg,  # TODO (Kirill): remove
        )
        criterion = ATSSCriterion(
            num_classes=num_classes,
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            loss_cls=CrossSigmoidFocalLoss(
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
        )
        model = SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            criterion=criterion,
            train_cfg=train_cfg,  # TODO (Kirill): remove
            test_cfg=test_cfg,  # TODO (Kirill): remove
        )
        model.init_weights()

        return model

    def _build_backbone(self, model_name: str) -> nn.Module:
        if "mobilenetv2" in model_name:
            from getitune.backend.lightning.models.common.backbones import build_model_including_pytorchcv

            return build_model_including_pytorchcv(
                cfg={
                    "type": "mobilenetv2_w1",
                    "out_indices": [2, 3, 4, 5],
                    "frozen_stages": -1,
                    "norm_eval": False,
                },
            )

        if "resnext101" in model_name:
            from getitune.backend.lightning.models.common.backbones import ResNeXt

            return ResNeXt(
                depth=101,
                groups=64,
                frozen_stages=1,
            )

        msg = f"Unknown backbone name: {model_name}"
        raise ValueError(msg)

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return DataInputParams(input_size=(800, 992), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
