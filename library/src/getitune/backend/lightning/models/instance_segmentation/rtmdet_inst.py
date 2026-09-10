# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RTMDetInst model implementations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, ClassVar, Literal

from torch import nn

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.losses import GIoULoss, QualityFocalLoss
from getitune.backend.lightning.models.common.utils.assigners import DynamicSoftLabelAssigner
from getitune.backend.lightning.models.common.utils.coders import DistancePointBBoxCoder
from getitune.backend.lightning.models.common.utils.prior_generators import MlvlPointGenerator
from getitune.backend.lightning.models.common.utils.samplers import PseudoSampler
from getitune.backend.lightning.models.detection.backbones import CSPNeXt
from getitune.backend.lightning.models.detection.detectors import SingleStageDetector
from getitune.backend.lightning.models.instance_segmentation.base import LightningInstanceSegModel
from getitune.backend.lightning.models.instance_segmentation.heads import RTMDetInstSepBNHead
from getitune.backend.lightning.models.instance_segmentation.losses import DiceLoss, RTMDetInstCriterion
from getitune.backend.lightning.models.instance_segmentation.necks import CSPNeXtPAFPN
from getitune.backend.lightning.models.modules.norm import build_norm_layer
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MaskRLEMeanAPFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import Tensor

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class RTMDetInst(LightningInstanceSegModel):
    """Implementation of RTMDetInst for instance segmentation.

    Args:
        label_info (LabelInfoTypes): Information about the labels used in the model.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
            If None is given, default parameters for the specific model will be used.
        model_name (str, optional): Name of the model. Defaults to "rtmdet_inst_tiny".
        optimizer (OptimizerCallable, optional): Optimizer for the model. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Scheduler for the model.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for evaluating the model.
            Defaults to MaskRLEMeanAPFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    _nms_always_embedded: ClassVar[bool] = True
    pretrained_urls: ClassVar[dict[str, str]] = {
        "rtmdet_inst_tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/"
        "rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["rtmdet_inst_tiny"] = "rtmdet_inst_tiny",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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

    def _create_model(self, num_classes: int | None = None) -> RTMDetInst:
        num_classes = num_classes if num_classes is not None else self.num_classes

        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.5},
            "score_thr": 0.05,
            "mask_thr_binary": 0.5,
            "max_per_img": 100,
            "min_bbox_size": 0,
            "nms_pre": 300,
        }

        backbone = CSPNeXt(model_name=self.model_name)
        neck = CSPNeXtPAFPN(model_name=self.model_name)
        bbox_head = RTMDetInstSepBNHead(
            num_classes=num_classes,
            in_channels=96,
            stacked_convs=2,
            share_conv=True,
            pred_kernel_size=1,
            feat_channels=96,
            normalization=partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
            activation=partial(nn.SiLU, inplace=True),
            anchor_generator=MlvlPointGenerator(
                offset=0,
                strides=[8, 16, 32],
            ),
            bbox_coder=DistancePointBBoxCoder(),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        criterion = RTMDetInstCriterion(
            num_classes=num_classes,
            loss_cls=QualityFocalLoss(
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_mask=DiceLoss(
                loss_weight=2.0,
                eps=5.0e-06,
                reduction="mean",
            ),
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
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {
                    "image": {0: "batch"},
                },
                "opset_version": 18,
                "autograd_inlining": False,
                "dynamo": False,
            },
            # TODO(Eugene): Add XAI support for RTMDetInst
            output_names=["boxes", "labels", "masks", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...]:
        """Forward function for export.

        NOTE : RTMDetInst uses explain_mode unlike other models.
        """
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(
            inputs,
            meta_info_list,
            explain_mode=self.explain_mode,
            with_nms=True,
        )

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return DataInputParams(input_size=(640, 640), mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
