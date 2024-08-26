# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDet model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from otx.algo.common.backbones import CSPNeXt
from otx.algo.common.losses import GIoULoss, QualityFocalLoss
from otx.algo.common.utils.assigners import DynamicSoftLabelAssigner
from otx.algo.common.utils.coders import DistancePointBBoxCoder
from otx.algo.common.utils.prior_generators import MlvlPointGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.detection.detectors import SingleStageDetector
from otx.algo.detection.heads import RTMDetSepBNHead
from otx.algo.detection.losses import RTMDetCriterion
from otx.algo.detection.necks import CSPNeXtPAFPN
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


PRETRAINED_ROOT: (
    str
) = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/"

PRETRAINED_WEIGHTS: dict[str, str] = {
    "rtmdet_tiny": PRETRAINED_ROOT + "rtmdet_tiny.pth",
}


class RTMDet(ExplainableOTXDetModel):
    """OTX Detection model class for RTMDet.

    Default input size per model:
        - rtmdet_tiny : (640, 640)
    """

    input_size_multiplier = 32
    mean: tuple[float, float, float] = (103.53, 116.28, 123.675)
    std: tuple[float, float, float] = (57.375, 57.12, 58.395)

    def __init__(
        self,
        model_name: Literal["rtmdet_tiny"],
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (640, 640),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.load_from: str = PRETRAINED_WEIGHTS[model_name]
        super().__init__(
            model_name=model_name,
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.001,
            "mask_thr_binary": 0.5,
            "max_per_img": 300,
            "min_bbox_size": 0,
            "nms_pre": 30000,
        }

        backbone = CSPNeXt(model_name=self.model_name)
        neck = CSPNeXtPAFPN(model_name=self.model_name)
        bbox_head = RTMDetSepBNHead(
            model_name=self.model_name,
            num_classes=num_classes,
            anchor_generator=MlvlPointGenerator(offset=0, strides=[8, 16, 32]),
            bbox_coder=DistancePointBBoxCoder(),
            train_cfg=train_cfg,  # TODO (sungchul, kirill): remove
            test_cfg=test_cfg,  # TODO (sungchul, kirill): remove
        )
        criterion = RTMDetCriterion(
            num_classes=num_classes,
            loss_cls=QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0),
            loss_bbox=GIoULoss(loss_weight=2.0),
        )
        return SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            criterion=criterion,
            train_cfg=train_cfg,  # TODO (sungchul, kirill): remove
            test_cfg=test_cfg,  # TODO (sungchul, kirill): remove
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
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
                "output_names": ["boxes", "labels"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(optimization_config={"preset": "mixed"})
