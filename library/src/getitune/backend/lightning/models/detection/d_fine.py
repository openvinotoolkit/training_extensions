# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""D-Fine model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.detection.backbones.hgnetv2 import HGNetv2
from getitune.backend.lightning.models.detection.detectors import DETR
from getitune.backend.lightning.models.detection.heads.dfine_decoder import DFINETransformer
from getitune.backend.lightning.models.detection.losses.dfine_loss import DFINECriterion
from getitune.backend.lightning.models.detection.necks.dfine_hybrid_encoder import HybridEncoder
from getitune.backend.lightning.models.detection.rtdetr import RTDETR
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class DFine(RTDETR):
    """getitune Detection model class for DFine.

    Attributes:
        pretrained_urls (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.
        input_size_multiplier (int): Multiplier for the input size.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
            If None, uses _default_preprocessing_params.
        model_name (literal, optional): Name of the model to use. Defaults to "dfine_hgnetv2_x".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MeanAveragePrecisionFMeasureCallable.
        multi_scale (bool, optional): Whether to use multi-scale training. Defaults to False.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    pretrained_urls: ClassVar[dict[str, str]] = {
        "dfine_hgnetv2_n": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth",
        "dfine_hgnetv2_s": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "dfine_hgnetv2_m": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
        "dfine_hgnetv2_l": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth",
        "dfine_hgnetv2_x": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth",
    }

    input_size_multiplier = 32

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "dfine_hgnetv2_n",
            "dfine_hgnetv2_s",
            "dfine_hgnetv2_m",
            "dfine_hgnetv2_l",
            "dfine_hgnetv2_x",
        ] = "dfine_hgnetv2_x",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,  # type: ignore[arg-type]
            label_info=label_info,
            data_input_params=data_input_params,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            multi_scale=multi_scale,
            torch_compile=torch_compile,
            tile_config=tile_config,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            export_nms=export_nms,
        )

    def _create_model(self, num_classes: int | None = None) -> DETR:
        num_classes = num_classes if num_classes is not None else self.num_classes
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        backbone = HGNetv2(model_name=self.model_name)
        encoder = HybridEncoder(model_name=self.model_name)
        decoder = DFINETransformer(
            model_name=self.model_name,
            num_classes=num_classes,
            eval_spatial_size=self.data_input_params.input_size,
        )
        criterion = DFINECriterion(
            weight_dict={
                "loss_vfl": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            alpha=0.75,
            gamma=2.0,
            reg_max=32,
            num_classes=num_classes,
        )

        backbone_lr_mapping = {
            "dfine_hgnetv2_n": 0.0004,
            "dfine_hgnetv2_s": 0.0001,
            "dfine_hgnetv2_m": 0.00002,
            "dfine_hgnetv2_l": 0.0000125,
            "dfine_hgnetv2_x": 0.0000125,
        }

        try:
            backbone_lr = backbone_lr_mapping[self.model_name]
        except KeyError as err:
            msg = f"Unsupported model name: {self.model_name}"
            raise ValueError(msg) from err

        optimizer_configuration = [
            # no weight decay for norm layers in backbone
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": backbone_lr},
            # lr for the backbone, but not norm layers is 0.00001
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": backbone_lr},
            # no weight decay for norm layers and biases in encoder and decoder layers
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        model = DETR(
            multi_scale=self.multi_scale,
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
            input_size=self.data_input_params.input_size[0],
        )
        model.init_weights()

        return model

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
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
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for D-FINE."""
        return {
            "model_type": "transformer",
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {"statistics_type": "QUANTILE", "aggregator_type": "MIN", "quantile_outlier_prob": 1e-4},
                    "max": {"statistics_type": "QUANTILE", "aggregator_type": "MAX", "quantile_outlier_prob": 1e-4},
                },
            },
        }

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary.

        If a RuntimeError occurs due to size mismatch, non-trainable anchors and valid_mask
        are removed from the checkpoint before loading.
        """
        try:
            return super().load_state_dict(ckpt, *args, **kwargs)
        except RuntimeError:
            # Remove non-trainable anchors and valid_mask from the checkpoint to avoid size mismatch
            ckpt.pop("model.decoder.anchors")
            ckpt.pop("model.decoder.valid_mask")
            return super().load_state_dict(ckpt, *args, strict=False, **kwargs)
