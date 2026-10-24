# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YOLOX model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from torch.export import Dim

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.losses import CrossEntropyLoss, IoULoss, L1Loss
from getitune.backend.lightning.models.detection.backbones import CSPDarknet
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors import SingleStageDetector
from getitune.backend.lightning.models.detection.heads import YOLOXHead
from getitune.backend.lightning.models.detection.losses import YOLOXCriterion
from getitune.backend.lightning.models.detection.necks import YOLOXPAFPN
from getitune.backend.lightning.models.detection.utils.assigners import SimOTAAssigner
from getitune.config.data import TileConfig
from getitune.data.entity.sample import SampleBatch
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


# YOLOX-S/L/X pretrained weights (MMDet) were trained on raw [0, 255] BGR images —
# no ImageNet normalization was applied during pretraining.
# These models are NOT compatible with 16-bit images because the uint8 pixel range
# assumption is baked into the weights.
_RAW_UINT8_MODELS: frozenset[str] = frozenset({"yolox_s", "yolox_l", "yolox_x"})


class YOLOX(LightningDetectionModel):
    """getitune Detection model class for YOLOX.

    Attributes:
        pretrained_urls (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): Parameters for image preprocessing.
            This parameter contains image input size, mean, and std, that is used to preprocess the input image.
            If None is given, default parameters for the specific model will be used.
            In most cases you don't need to set this parameter unless you change the image size or pretrained weights.
            Defaults to None.
        model_name (str, optional): Name of the model to use. Defaults to "yolox_s".
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
        "yolox_tiny": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/"
        "object_detection/v2/yolox_tiny_8x8.pth",
        "yolox_s": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/"
        "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
        "yolox_l": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/"
        "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        "yolox_x": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/"
        "yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
    }

    input_size_multiplier = 32

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["yolox_tiny", "yolox_s", "yolox_l", "yolox_x"] = "yolox_s",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
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

        # Raw uint8 models expect [0, 255] inputs; reject 16-bit data.
        if (
            model_name in _RAW_UINT8_MODELS
            and self.data_input_params.intensity_config is not None
            and self.data_input_params.intensity_config.storage_dtype in ("uint16", "int16")
        ):
            msg = (
                f"YOLOX ({model_name}) does not support high-bit-depth inputs "
                f"(got storage_dtype='{self.data_input_params.intensity_config.storage_dtype}'). "
                "Use yolox_tiny or a model with normalization for 16-bit images."
            )
            raise ValueError(msg)

    def _create_model(self, num_classes: int | None = None) -> SingleStageDetector:
        num_classes = num_classes if num_classes is not None else self.num_classes
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.01,
            "max_per_img": 100,
        }
        backbone = CSPDarknet(model_name=self.model_name)
        neck = YOLOXPAFPN(model_name=self.model_name)
        bbox_head = YOLOXHead(
            model_name=self.model_name,
            num_classes=num_classes,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        criterion = YOLOXCriterion(
            num_classes=num_classes,
            loss_cls=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_bbox=IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
            loss_obj=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_l1=L1Loss(reduction="sum", loss_weight=1.0),
        )
        model = SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            criterion=criterion,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        model.init_weights()

        return model

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        resize_mode: Literal["standard", "fit_to_window_letterbox"] = "fit_to_window_letterbox"
        if self.tile_config.enable_tiler:
            resize_mode = "standard"
        swap_rgb = self.model_name != "yolox_tiny"  # only YOLOX-TINY uses RGB

        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode=resize_mode,
            pad_value=114,
            swap_rgb=swap_rgb,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "export_params": True,
                "opset_version": 18,
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "keep_initializers_as_inputs": False,
                "verbose": False,
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: ExportFormat,
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export this model to the specified output directory.

        This is required to patch getitune.algo.detection.backbones.csp_darknet.Focus.forward to export forward.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (ExportFormat): format of the output model
            precision (Precision): precision of the output model

        Returns:
            Path: path to the exported model.
        """
        # patch getitune.algo.detection.backbones.csp_darknet.Focus.forward
        orig_focus_forward = self.model.backbone.stem.forward
        try:
            self.model.backbone.stem.forward = self.model.backbone.stem.export
            return super().export(output_dir, base_name, export_format, precision)
        finally:
            self.model.backbone.stem.forward = orig_focus_forward

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        # YOLOX s/l/x: std=1/255 encodes the x*255 rescaling so that ModelAPI
        # applies (x - 0) / (1/255) = x * 255 during inference, producing the
        # [0, 255] float range the pretrained weights expect.
        _zero_mean = (0.0, 0.0, 0.0)
        _scale_to_255_std = (1 / 255, 1 / 255, 1 / 255)
        # ImageNet normalization in 0-1 range (standard convention for models with intensity scaling)
        _imagenet_mean = (0.485, 0.456, 0.406)
        _imagenet_std = (0.229, 0.224, 0.225)

        return {
            # YOLOX-tiny uses ImageNet normalization (0-1 range, same as DEIM/ViT/EfficientNet)
            "yolox_tiny": DataInputParams(
                input_size=(640, 640),
                mean=_imagenet_mean,
                std=_imagenet_std,
            ),
            # YOLOX s/l/x: std=1/255 combined with intensity scale_to_unit (propagated
            # from SubsetConfig at runtime) gives u8->/255->[0,1]->x*255->[0,255] at inference.
            "yolox_s": DataInputParams(
                input_size=(640, 640),
                mean=_zero_mean,
                std=_scale_to_255_std,
            ),
            "yolox_l": DataInputParams(
                input_size=(640, 640),
                mean=_zero_mean,
                std=_scale_to_255_std,
            ),
            "yolox_x": DataInputParams(
                input_size=(640, 640),
                mean=_zero_mean,
                std=_scale_to_255_std,
            ),
        }

    def _customize_inputs(self, entity: SampleBatch) -> dict[str, Any]:
        if self.model_name in _RAW_UINT8_MODELS and entity.imgs_info is not None:
            for info in entity.imgs_info:
                if info is not None and getattr(info, "bit_depth", 8) > 8:
                    msg = (
                        f"YOLOX ({self.model_name}) does not support images with bit_depth > 8. "
                        f"Got bit_depth={info.bit_depth}. "
                        "Pretrained weights require [0, 255] uint8-range inputs. "
                        "Use yolox_tiny or a model with normalization for 16-bit images."
                    )
                    raise RuntimeError(msg)

        return super()._customize_inputs(entity)
