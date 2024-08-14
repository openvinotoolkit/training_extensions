# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMPose model implementations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from otx.algo.common.backbones import CSPNeXt
from otx.algo.keypoint_detection.heads.rtmcc_head import RTMCCHead
from otx.algo.keypoint_detection.losses.kl_discret_loss import KLDiscretLoss
from otx.algo.keypoint_detection.topdown import TopdownPoseEstimator
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.pck import PCKMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.keypoint_detection import OTXKeypointDetectionModel
from torch import nn

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from otx.core.exporter.base import OTXModelExporter
    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.export import TaskLevelExportParameters
    from otx.core.types.label import LabelInfoTypes


class RTMPose(OTXKeypointDetectionModel):
    """OTX keypoint detection model class for RTMPose."""

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Exporter should have a input_size but it is given by {self.input_size}"
            raise ValueError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=True,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["points"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "points": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
            },
            output_names=["points", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(optimization_config={"preset": "mixed"})


class RTMPoseTiny(RTMPose):
    """RTMPose Tiny Model."""

    load_from = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth"
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (256, 192),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = PCKMeasureCallable,
        torch_compile: bool = False,
    ) -> None:
        self.mean = (0.0, 0.0, 0.0)
        self.std = (255.0, 255.0, 255.0)
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _build_model(self, num_classes: int) -> RTMPose:
        simcc_split_ratio = 2.0
        sigma = (4.9, 5.66)

        backbone = CSPNeXt(
            arch="P5",
            expand_ratio=0.5,
            deepen_factor=0.167,
            widen_factor=0.375,
            out_indices=(4,),
            channel_attention=True,
            normalization=nn.BatchNorm2d,
            activation=partial(nn.SiLU, inplace=True),
        )
        head = RTMCCHead(
            out_channels=num_classes,
            in_channels=384,
            input_size=self.input_size,
            in_featuremap_size=(self.input_size[0] // 32, self.input_size[1] // 32),
            simcc_split_ratio=simcc_split_ratio,
            final_layer_kernel_size=7,
            loss=KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True),
            decoder_cfg={
                "input_size": self.input_size,
                "simcc_split_ratio": simcc_split_ratio,
                "sigma": sigma,
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

        return TopdownPoseEstimator(
            backbone=backbone,
            head=head,
        )
