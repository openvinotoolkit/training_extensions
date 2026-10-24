# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter model implementation for instance segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.edgecrafter_mixin import EdgeCrafterMixin
from getitune.backend.lightning.models.detection.detectors.edgecrafter import ECDETRDetector
from getitune.backend.lightning.models.instance_segmentation.base import LightningInstanceSegModel
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MaskRLEMeanAPFMeasureCallable
from getitune.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class EdgeCrafterInst(EdgeCrafterMixin, LightningInstanceSegModel):  # pyrefly: ignore[inconsistent-inheritance]
    """getitune Instance Segmentation model class for EdgeCrafter.

    Extends the EdgeCrafter detection backbone with a dedicated instance-segmentation
    head (``SegmentationHead`` inside :class:`ECTransformer`).  The backbone is
    initialised from the ECSeg pretrained weights
    (``ecseg_vitt/vittplus/vits/vitsplus``) which are trained jointly for
    detection and segmentation.

    Original repository: https://github.com/Intellindust-AI-Lab/EdgeCrafter
    Paper: https://arxiv.org/abs/2603.18739

    Args:
        label_info: Information about the labels.
        data_input_params: Preprocessing parameters (input size, mean, std).
            When ``None`` the ``_default_preprocessing_params`` dict is used.
        model_name: One of ``"edgecrafter_s"``, ``"edgecrafter_m"``,
            ``"edgecrafter_l"``, ``"edgecrafter_x"``.
        optimizer: Optimizer callable. Defaults to :data:`DefaultOptimizerCallable`.
        scheduler: LR-scheduler callable. Defaults to :data:`DefaultSchedulerCallable`.
        metric: Metric callable. Defaults to mask RLE mean AP.
        multi_scale: Whether to use multi-scale training. Defaults to ``False``.
        backbone_lr: Learning rate for the backbone. Defaults to the upstream
            per-variant value in :class:`EdgeCrafterMixin`.
        torch_compile: Whether to use ``torch.compile``. Defaults to ``False``.
        tile_config: Tiling configuration. Defaults to disabled tiler.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    # ECSeg weights - loaded in place of ECDet weights.
    pretrained_urls: ClassVar[dict[str, str]] = {
        "edgecrafter_s": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_s.pth",
        "edgecrafter_m": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_m.pth",
        "edgecrafter_l": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth",
        "edgecrafter_x": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_x.pth",
    }

    # Instance-segmentation loss weights + matcher costs (adds mask terms on top of detection).
    _loss_weights: ClassVar[dict[str, float]] = {
        "loss_mal": 2.0,
        "loss_bbox": 1.0,
        "loss_giou": 1.0,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
        "loss_mask_ce": 5.0,
        "loss_mask_dice": 5.0,
    }
    _matcher_cost_dict: ClassVar[dict[str, int | float]] = {
        "cost_class": 2,
        "cost_bbox": 1,
        "cost_giou": 1,
        "cost_mask": 5,
        "cost_dice": 5,
    }
    _mask_downsample_ratio: ClassVar[int] = 4
    _backbone_key: ClassVar[str] = "seg_backbone_name"

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
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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
        """Create EdgeCrafter instance segmentation model.

        Args:
            num_classes: Number of classes. Defaults to ``self.num_classes``.

        Returns:
            Configured :class:`ECDETRDetector` with instance-segmentation head.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes
        return self._build_ec_model(num_classes, backbone_lr=self.backbone_lr)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Export parameters for EdgeCrafter instance segmentation.

        Uses the ``DETRInstSeg`` model type which triggers full-image mask
        postprocessing in ModelAPI.  A conservative IoU threshold (0.8) suppresses
        only near-duplicate predictions.
        """
        return super()._export_parameters.wrap(model_type="DETRInstSeg", iou_threshold=0.8)

    @property
    def _exporter(self) -> ModelExporter:
        """Creates :class:`ModelExporter` for EdgeCrafter instance segmentation."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {"images": {0: "batch"}},
                "dynamo": False,
                "do_constant_folding": True,
                "opset_version": 18,
            },
            output_names=["boxes", "labels", "masks"],
        )

    def forward_for_tracing(self, inputs: torch.Tensor) -> dict[str, Any]:
        """Forward pass used for ONNX / OpenVINO export.

        Reformats the deploy-mode output into the MaskRCNN-compatible format:
        ``boxes`` [B, Q, 5] (x1,y1,x2,y2,score in pixel coords),
        ``labels`` [B, Q], ``masks`` [B, Q, H, W].

        Args:
            inputs: Image batch [B, C, H, W].

        Returns:
            Dict with ``boxes``, ``labels``, ``masks``.
        """
        meta_info_list = self._default_is_meta(inputs)
        result = self.model.export(inputs, meta_info_list, with_nms=self.export_nms)  # type: ignore[attr-defined]  # pyrefly: ignore[not-callable]
        return self._make_is_export_prediction(inputs, result)

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Default preprocessing parameters for each EdgeCrafter instance-seg variant."""
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        return {
            "edgecrafter_s": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_m": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_l": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_x": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
        }
