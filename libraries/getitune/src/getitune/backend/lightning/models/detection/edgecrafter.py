# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter model implementation for object detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, cast

import torch
from torch import Tensor

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.edgecrafter_mixin import EdgeCrafterMixin
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors.edgecrafter import ECDETRDetector
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class EdgeCrafter(EdgeCrafterMixin, LightningDetectionModel):  # pyrefly: ignore[inconsistent-inheritance]
    """getitune Detection model class for EdgeCrafter.

    EdgeCrafter is a family of compact ViT-based DETR models designed for edge
    deployment via task-specialised distillation.  Four sizes are available:
    S (small), M (medium), L (large), and X (extra-large).

    Original repository: https://github.com/Intellindust-AI-Lab/EdgeCrafter
    Paper: https://arxiv.org/abs/2603.18739

    The model should be used with
    :class:`~getitune.backend.lightning.callbacks.aug_scheduler.AdaptiveTrainScheduling`
    and optional augmentation scheduling callbacks.

    Attributes:
        pretrained_urls: Mapping from model-name to pretrained weight URL.
        input_size_multiplier: Patch-size aligned input divisor (16).

    Args:
        label_info: Information about the labels.
        data_input_params: Preprocessing parameters (input size, mean, std).
            When ``None`` the ``_default_preprocessing_params`` dict is used.
        model_name: One of ``"edgecrafter_{s,m,l,x}"``.
        optimizer: Optimizer callable. Defaults to :data:`DefaultOptimizerCallable`.
        scheduler: LR-scheduler callable. Defaults to :data:`DefaultSchedulerCallable`.
        metric: Metric callable. Defaults to MAP F-measure.
        multi_scale: Whether to use multi-scale training. Defaults to ``False``.
        backbone_lr: Learning rate for the backbone. Defaults to the upstream
            per-variant value in :class:`EdgeCrafterMixin`.
        torch_compile: Whether to use ``torch.compile``. Defaults to ``False``.
        tile_config: Tiling configuration. Defaults to disabled tiler.
                pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    pretrained_urls: ClassVar[dict[str, str]] = {
        "edgecrafter_s": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth",
        "edgecrafter_m": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_m.pth",
        "edgecrafter_l": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth",
        "edgecrafter_x": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_x.pth",
    }

    # Detection loss weights (matcher/backbone-key/mask-ratio use EdgeCrafterMixin defaults).
    _loss_weights: ClassVar[dict[str, float]] = {
        "loss_mal": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }

    input_size_multiplier = 16

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "edgecrafter_s",
            "edgecrafter_m",
            "edgecrafter_l",
            "edgecrafter_x",
        ] = "edgecrafter_s",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        backbone_lr: float | None = None,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        self.multi_scale = multi_scale
        self.backbone_lr = backbone_lr
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

    def _create_model(self, num_classes: int | None = None) -> ECDETRDetector:
        """Create EdgeCrafter detection model.

        Args:
            num_classes: Number of classes. Defaults to ``self.num_classes``.

        Returns:
            Configured :class:`ECDETRDetector`.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes
        return self._build_ec_model(num_classes, backbone_lr=self.backbone_lr)

    def forward_for_tracing(self, inputs: Tensor) -> dict[str, Tensor]:
        """Export-mode forward.

        The detector returns boxes in pixel coordinates, but ModelAPI expects
        DETR-style normalized ``xyxy`` boxes (like DEIM/D-FINE).  Normalize the
        exported boxes by the input spatial size here.
        """
        result = cast("dict[str, Tensor]", super().forward_for_tracing(inputs))
        _, _, height, width = inputs.shape
        scale = torch.tensor([width, height, width, height], device=inputs.device, dtype=inputs.dtype)
        result["bboxes"] = result["bboxes"] / scale
        return result

    @property
    def _exporter(self) -> ModelExporter:
        """Creates :class:`ModelExporter` for EdgeCrafter detection."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {"images": {0: "batch"}},
                "dynamo": True,
                "opset_version": 18,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Default preprocessing parameters for each EdgeCrafter variant."""
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        return {
            "edgecrafter_s": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_m": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_l": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_x": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
        }
