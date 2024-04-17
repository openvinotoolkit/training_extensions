# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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
from otx.algo.segmentation.heads import CustomLightHamHead
from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class _SegNext(nn.Module):
    def __init__(self,
                 num_classes,
                 in_chans=3,
                 embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 dropout_ratio=0.1,
                 drop_path_rate=0.,
                 depths=[3, 3, 12, 3],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 in_channels=[128,320,512],
                 ham_channels=512,
                 channels=512,
                 spatial=True,
                 MD_S=1,   #MD_S
                 MD_R=64,  #MD_R
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 eta=0.9,
                 ignore_index=255,
                 checkpoint=None):
        super().__init__()

        self.backbone = MSCAN(in_channels=in_chans, embed_dims=embed_dims, mlp_ratios=mlp_ratios,
                             drop_rate=dropout_ratio, drop_path_rate=drop_path_rate, depths=depths,
                             num_stages=num_stages, norm_cfg=norm_cfg)
        self.decode_head = CustomLightHamHead(in_channels=in_channels, ham_channels=ham_channels, num_classes=num_classes, in_index=[1,2,3], channels=channels, ham_kwargs=dict(MD_S=MD_S, MD_R=MD_R, train_steps=train_steps,
                                    eval_steps=eval_steps, inv_t=inv_t))
        self.conv_seg = nn.Sequential(nn.Conv2d(channels, num_classes, kernel_size=1), nn.ReLU(inplace=True))
        self.criterion = CrossEntropyLossWithIgnore()
        self.init_weights()
        if checkpoint:
            checkpoint = load_url(checkpoint)
            # checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        #self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None


    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        outputs = self.conv_seg(feat)

        return outputs

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
        # outputs = self.cls_seg(dec_out)  # here output will be B x C x H/8 x W/8
        outputs = F.interpolate(outputs, size=images.size()[-2:], mode='bilinear', align_corners=True)

        if self.training:
            return self.criterion(outputs, masks)

        return outputs


class SegNext(OTXSegmentationModel):
    """SegNext Model."""

    def __init__(
        self,
        num_classes: int,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        checkpoint: str = None,
        # configuration: dict[str, Any] = {},
    ) -> None:
        # self.num_classes = num_classes
        # self.config = configuration
        self.checkpoint = checkpoint
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        return _SegNext(num_classes=self.num_classes, checkpoint=self.checkpoint)

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
            masks.append(output)
        breakpoint()
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
