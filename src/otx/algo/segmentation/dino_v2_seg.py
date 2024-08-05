# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from otx.algo.segmentation.backbones import DinoVisionTransformer
from otx.algo.segmentation.heads import FCNHead
from otx.core.model.segmentation import TorchVisionCompatibleModel
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.metrics.dice import SegmCallable

from .base_model import BaseSegmModel

if TYPE_CHECKING:
    from torch import nn
    from typing_extensions import Self
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes
    from otx.core.metrics import MetricCallable


class DinoV2Seg(BaseSegmModel):
    """DinoV2Seg Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "name": "dinov2_vits14",
        "freeze_backbone": True,
        "out_index": [8, 9, 10, 11],
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"type": "SyncBN", "requires_grad": True},
        "in_channels": [384, 384, 384, 384],
        "in_index": [0, 1, 2, 3],
        "input_transform": "resize_concat",
        "channels": 1536,
        "kernel_size": 1,
        "num_convs": 1,
        "concat_input": False,
        "dropout_ratio": -1,
        "align_corners": False,
        "pretrained_weights": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_linear_head.pth",
    }


class OTXDinoV2Seg(TorchVisionCompatibleModel):
    """DinoV2Seg Model."""
    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: Sequence[int] = (1, 3, 560, 560),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        backbone_configuration: dict[str, Any] | None = None,
        decode_head_configuration: dict[str, Any] | None = None,
        criterion_configuration: list[dict[str, Any]] | None = None,
        export_image_configuration: dict[str, Any] | None = None,
        name_base_model: str = "semantic_segmentation_model",
    ):
        if input_size[-1] % 14 != 0 or input_size[-2] % 14 != 0:
            msg = f"Input size should be a multiple of 14, but got {input_size[-2:]} instead."
            raise ValueError(msg)

        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            backbone_configuration=backbone_configuration,
            decode_head_configuration=decode_head_configuration,
            criterion_configuration=criterion_configuration,
            export_image_configuration=export_image_configuration,
            name_base_model=name_base_model,
        )

    def _create_model(self) -> nn.Module:
        # merge configurations with defaults overriding them
        backbone_configuration = DinoV2Seg.default_backbone_configuration | self.backbone_configuration
        decode_head_configuration = DinoV2Seg.default_decode_head_configuration | self.decode_head_configuration
        backbone = DinoVisionTransformer(**backbone_configuration)
        decode_head = FCNHead(num_classes=self.num_classes, **decode_head_configuration)
        return DinoV2Seg(
            backbone=backbone,
            decode_head=decode_head,
            criterion_configuration=self.criterion_configuration,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}

    def to(self, *args, **kwargs) -> Self:
        """Return a model with specified device."""
        ret = super().to(*args, **kwargs)
        if self.device.type == "xpu":
            msg = f"{type(self).__name__} doesn't support XPU."
            raise RuntimeError(msg)
        return ret
