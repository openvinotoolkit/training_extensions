# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Dict

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.dice import SegmCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.segmentation import MMSegCompatibleModel, OTXSegmentationModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.utils.utils import get_mean_std_from_data_processing
from typing import TYPE_CHECKING, Any
import torch
from torch import nn
import math
import torch.nn.functional as F
from torchvision import tv_tensors
from torch.utils.model_zoo import load_url

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.metrics.dice import SegmCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.schedulers import LRSchedulerListCallable
from otx.algo.segmentation.backbones import MSCAN
# from otx.algo.segmentation.heads import LightHamHead
from otx.algo.segmentation.losses import create_criterion
from mmseg.models.decode_heads import LightHamHead

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class SegNext(nn.Module):
    def __init__(
        self,
        backbone: MSCAN,
        decode_head: LightHamHead,
        criterion_configuration: Dict[str, str | Any] = {"type": "CrossEntropyLoss", "ignore_index": 255},
        pretrained_weights: str | None = None,
    ) -> None:
        """
        Initializes a SegNext model.

        Args:
            backbone (MSCAN): The backbone of the model.
            decode_head (LightHamHead): The decode head of the model.
            criterion (Dict[str, Union[str, int]]): The criterion of the model.
                Defaults to {"type": "CrossEntropyLoss", "ignore_index": 255}.
            pretrained_weights (Optional[str]): The path to the pretrained weights.
                Defaults to None.

        Returns:
            None
        """
        super().__init__()

        self.backbone = backbone
        self.decode_head = decode_head
        self.criterion = create_criterion(**criterion_configuration)
        self.init_weights()

        if pretrained_weights:
            # load pretrained weights
            pretrained_weights = load_url(pretrained_weights)
            self.load_state_dict(pretrained_weights['state_dict'], strict=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)

    def forward(self, images, masks):
        enc_feats = self.backbone(images)
        outputs = self.decode_head(enc_feats)
        outputs = F.interpolate(outputs, size=images.size()[-2:], mode='bilinear', align_corners=True)

        if self.training:
            return self.criterion(outputs, masks)

        return outputs


class OTXSegNext(OTXSegmentationModel):
    """SegNext Model."""

    def __init__(
        self,
        num_classes: int,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        pretrained_weights: str = None,
        backbone_configuration: dict[str, Any] = {},
        decode_head_configuration: dict[str, Any] = {},
        criterion_configuration: dict[str, Any] = {}
    ) -> None:
        # self.num_classes = num_classes
        self.backbone_configuration = backbone_configuration
        self.decode_head_configuration = decode_head_configuration
        self.criterion_configuration = criterion_configuration
        self.pretrained_weights = pretrained_weights
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        backbone = MSCAN(**self.backbone_configuration)
        decode_head = LightHamHead(num_classes=self.num_classes, **self.decode_head_configuration)
        return SegNext(
            backbone=backbone,
            decode_head=decode_head,
            pretrained_weights=self.pretrained_weights,
            criterion_configuration = self.criterion_configuration
        )

    def _customize_inputs(self, entity: SegBatchDataEntity) -> dict[str, Any]:
        masks = torch.stack(entity.masks).long()
        inputs = {"images": entity.images, "masks" : masks}
        return inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SegBatchDataEntity,
    ) -> SegBatchPredEntity | OTXBatchLossEntity:

        if self.training:
            losses = OTXBatchLossEntity()
            losses["loss"] = outputs
            return losses

        masks = []

        for output in outputs:
            masks.append(output.argmax(dim=0))

        return SegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
        )

    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        mean, std = get_mean_std_from_data_processing(self.config)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=None,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_segnext_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for SegNext."""
        # TODO(Kirill): check PTQ removing hamburger from ignored_scope
        return {
            "ignored_scope": {
                "patterns": ["__module.decode_head.hamburger*"],
                "types": [
                    "Add",
                    "MVN",
                    "Divide",
                    "Multiply",
                ],
            },
        }


# class MMsegSegNext(MMSegCompatibleModel):
#     """SegNext Model."""

#     def __init__(
#         self,
#         num_classes: int,
#         variant: Literal["b", "s", "t"],
#         optimizer: OptimizerCallable = DefaultOptimizerCallable,
#         scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
#         metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
#         torch_compile: bool = False,
#     ) -> None:
#         model_name = f"segnext_{variant}"
#         config = read_mmconfig(model_name=model_name)
#         super().__init__(
#             num_classes=num_classes,
#             config=config,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             metric=metric,
#             torch_compile=torch_compile,
#         )

#     @property
#     def _exporter(self) -> OTXModelExporter:
#         """Creates OTXModelExporter object that can export the model."""
#         mean, std = get_mean_std_from_data_processing(self.config)

#         return OTXNativeModelExporter(
#             task_level_export_parameters=self._export_parameters,
#             input_size=self.image_size,
#             mean=mean,
#             std=std,
#             resize_mode="standard",
#             pad_value=0,
#             swap_rgb=False,
#             via_onnx=False,
#             onnx_export_configuration=None,
#             output_names=None,
#         )

#     def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
#         """Load the previous OTX ckpt according to OTX2.0."""
#         return OTXv1Helper.load_seg_segnext_ckpt(state_dict, add_prefix)

#     @property
#     def _optimization_config(self) -> dict[str, Any]:
#         """PTQ config for SegNext."""
#         # TODO(Kirill): check PTQ removing hamburger from ignored_scope
#         return {
#             "ignored_scope": {
#                 "patterns": ["__module.decode_head.hamburger*"],
#                 "types": [
#                     "Add",
#                     "MVN",
#                     "Divide",
#                     "Multiply",
#                 ],
#             },
#         }
