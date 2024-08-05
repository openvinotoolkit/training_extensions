# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""
from __future__ import annotations

import torch
import torchvision
import copy
from typing import TYPE_CHECKING, Any, ClassVar

from otx.algo.segmentation.backbones import MSCAN
from otx.algo.segmentation.heads import LightHamHead
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.segmentation import TorchVisionCompatibleModel
from otx.algo.segmentation.mean_teacher import MeanTeacher
from otx.core.data.entity.segmentation import SegBatchDataEntity

from .base_model import BaseSegmModel

if TYPE_CHECKING:
    from torch import nn


class SegNextB(BaseSegmModel):
    """SegNextB Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "act_cfg": {"type": "GELU"},
        "attention_kernel_paddings": [2, [0, 3], [0, 5], [0, 10]],
        "attention_kernel_sizes": [5, [1, 7], [1, 11], [1, 21]],
        "depths": [3, 3, 12, 3],
        "drop_path_rate": 0.1,
        "drop_rate": 0.0,
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "norm_cfg": {"requires_grad": True, "type": "BN"},
        "pretrained_weights": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth",
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "ham_kwargs": {"md_r": 16, "md_s": 1, "eval_steps": 7, "train_steps": 6},
        "in_channels": [128, 320, 512],
        "in_index": [1, 2, 3],
        "norm_cfg": {"num_groups": 32, "requires_grad": True, "type": "GN"},
        "align_corners": False,
        "channels": 512,
        "dropout_ratio": 0.1,
        "ham_channels": 512,
    }


class SegNextS(BaseSegmModel):
    """SegNextS Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "act_cfg": {"type": "GELU"},
        "attention_kernel_paddings": [2, [0, 3], [0, 5], [0, 10]],
        "attention_kernel_sizes": [5, [1, 7], [1, 11], [1, 21]],
        "depths": [2, 2, 4, 2],
        "drop_path_rate": 0.1,
        "drop_rate": 0.0,
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "norm_cfg": {"requires_grad": True, "type": "BN"},
        "pretrained_weights": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth",
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"num_groups": 32, "requires_grad": True, "type": "GN"},
        "ham_kwargs": {"md_r": 16, "md_s": 1, "eval_steps": 7, "rand_init": True, "train_steps": 6},
        "in_channels": [128, 320, 512],
        "in_index": [1, 2, 3],
        "align_corners": False,
        "channels": 256,
        "dropout_ratio": 0.1,
        "ham_channels": 256,
    }


class SegNextT(BaseSegmModel):
    """SegNextT Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "act_cfg": {"type": "GELU"},
        "attention_kernel_paddings": [2, [0, 3], [0, 5], [0, 10]],
        "attention_kernel_sizes": [5, [1, 7], [1, 11], [1, 21]],
        "depths": [3, 3, 5, 2],
        "drop_path_rate": 0.1,
        "drop_rate": 0.0,
        "embed_dims": [32, 64, 160, 256],
        "mlp_ratios": [8, 8, 4, 4],
        "norm_cfg": {"requires_grad": True, "type": "BN"},
        "pretrained_weights": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth",
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "ham_kwargs": {"md_r": 16, "md_s": 1, "eval_steps": 7, "rand_init": True, "train_steps": 6},
        "norm_cfg": {"num_groups": 32, "requires_grad": True, "type": "GN"},
        "in_channels": [64, 160, 256],
        "in_index": [1, 2, 3],
        "align_corners": False,
        "channels": 256,
        "dropout_ratio": 0.1,
        "ham_channels": 256,
    }


SEGNEXT_VARIANTS = {
    "SegNextB": SegNextB,
    "SegNextS": SegNextS,
    "SegNextT": SegNextT,
}


class OTXSegNext(TorchVisionCompatibleModel):
    """SegNext Model."""

    def _create_model(self) -> nn.Module:
        segnext_model_class = SEGNEXT_VARIANTS[self.name_base_model]
        # merge configurations with defaults overriding them
        backbone_configuration = segnext_model_class.default_backbone_configuration | self.backbone_configuration
        decode_head_configuration = (
            segnext_model_class.default_decode_head_configuration | self.decode_head_configuration
        )
        # initialize backbones
        backbone = MSCAN(**backbone_configuration)
        decode_head = LightHamHead(num_classes=self.num_classes, **decode_head_configuration)
        return segnext_model_class(
            backbone=backbone,
            decode_head=decode_head,
            criterion_configuration=self.criterion_configuration,
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
                "patterns": ["__module.model.decode_head.hamburger*"],
                "types": [
                    "Add",
                    "MVN",
                    "Divide",
                    "Multiply",
                ],
            },
        }

class SemiSLSegNext(OTXSegNext):
    """SegNext Model."""

    def _customize_inputs(self, entity: SegBatchDataEntity) -> dict[str, Any]:

        if not isinstance(entity, dict):
            if self.training:
                msg = "unlabeled inputs should be provided for semi-sl training"
                raise RuntimeError(msg)
            return super()._customize_inputs(entity)

        entity["labeled"].masks = torch.stack(entity["labeled"].masks).long()
        w_u_images = entity["weak_transforms"].images
        s_u_images = entity["strong_transforms"].images
        unlabeled_img_metas = entity["weak_transforms"].imgs_info
        labeled_inputs = entity["labeled"]

        return {"inputs": labeled_inputs.images,
                "unlabeled_weak_images": w_u_images,
                "unlabeled_strong_images": s_u_images,
                "global_step": self.trainer.global_step,
                "steps_per_epoch": self.trainer.num_training_batches,
                "img_metas": labeled_inputs.imgs_info,
                "unlabeled_img_metas": unlabeled_img_metas,
                "masks": labeled_inputs.masks,
                "mode": "loss"}

    def _create_model(self) -> nn.Module:
        segnext_model_class = SEGNEXT_VARIANTS[self.name_base_model]
        # merge configurations with defaults overriding them
        backbone_configuration = segnext_model_class.default_backbone_configuration | self.backbone_configuration
        decode_head_configuration = (
            segnext_model_class.default_decode_head_configuration | self.decode_head_configuration
        )
        # initialize backbones
        backbone = MSCAN(**backbone_configuration)
        decode_head = LightHamHead(num_classes=self.num_classes, **decode_head_configuration)
        base_model = segnext_model_class(
            backbone=backbone,
            decode_head=decode_head,
            criterion_configuration=self.criterion_configuration,
        )

        return MeanTeacher(base_model, unsup_weight=0.5, drop_unrel_pixels_percent=20, semisl_start_epoch=1)
