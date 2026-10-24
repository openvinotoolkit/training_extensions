# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RTMPose model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.detection.backbones import CSPNeXt
from getitune.backend.lightning.models.keypoint_detection.base import LightningKeypointDetectionModel
from getitune.backend.lightning.models.keypoint_detection.detectors.topdown import TopdownPoseEstimator
from getitune.backend.lightning.models.keypoint_detection.heads.rtmcc_head import RTMCCHead
from getitune.backend.lightning.models.keypoint_detection.losses.kl_discret_loss import KLDiscretLoss
from getitune.metrics.pck import PCKMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from getitune.backend.lightning.exporter.base import ModelExporter
    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.export import TaskLevelExportParameters
    from getitune.types.label import LabelInfoTypes


class RTMPose(LightningKeypointDetectionModel):
    """RTMPose Model."""

    pretrained_urls: ClassVar[dict[str, str]] = {
        "rtmpose_tiny": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth",
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["rtmpose_tiny"] = "rtmpose_tiny",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = PCKMeasureCallable,
        torch_compile: bool = False,
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
        )

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        backbone = CSPNeXt(model_name=self.model_name)
        head = RTMCCHead(
            out_channels=num_classes,
            in_channels=384,
            input_size=self.data_input_params.input_size,
            in_featuremap_size=(self.data_input_params.input_size[0] // 32, self.data_input_params.input_size[1] // 32),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            loss=KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True),
            decoder_cfg={
                "input_size": self.data_input_params.input_size,
                "simcc_split_ratio": 2.0,
                "sigma": (4.9, 5.66),
                "normalize": False,
                "use_dark": False,
            },
            gau_cfg={
                "num_token": num_classes,
                "in_token_dims": 256,
                "out_token_dims": 256,
                "s": 128,
                "expansion_factor": 2,
                "dropout_rate": 0.0,
                "drop_path": 0.0,
                "act_fn": "SiLU",
                "use_rel_bias": False,
                "pos_enc": False,
            },
        )

        model = TopdownPoseEstimator(
            backbone=backbone,
            head=head,
        )
        model.init_weights()

        return model

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        if self.explain_mode:
            msg = "Export with explain is not supported for RTMPose model."
            raise ValueError(msg)

        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="fit_to_window",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "dynamic_axes": {
                    "image": {0: "batch"},
                    "pred_x": {0: "batch"},
                    "pred_y": {0: "batch"},
                },
                "autograd_inlining": False,
                "opset_version": 18,
            },
            output_names=["pred_x", "pred_y"],
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(optimization_config={"preset": "mixed"})
