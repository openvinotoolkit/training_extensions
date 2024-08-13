# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""
from __future__ import annotations

<<<<<<< HEAD
from typing import TYPE_CHECKING, Any, ClassVar

from otx.algo.segmentation.backbones import MSCAN
from otx.algo.segmentation.heads import LightHamHead
from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore
from otx.algo.segmentation.segmentors import BaseSegmModel
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.segmentation import OTXSegmentationModel

if TYPE_CHECKING:
    from torch import nn

=======
from typing import Any, ClassVar

import torch
from torch import nn

from otx.algo.segmentation.backbones import MSCAN
from otx.algo.segmentation.heads import LightHamHead
from otx.algo.segmentation.segmentors import BaseSegmModel, MeanTeacher
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.segmentation import SegBatchDataEntity
from otx.core.model.segmentation import TorchVisionCompatibleModel

>>>>>>> develop

class SegNext(OTXSegmentationModel):
    """SegNext Model."""

    AVAILABLE_MODEL_VERSIONS: ClassVar[list[str]] = [
        "segnext_tiny",
        "segnext_small",
        "segnext_base",
    ]

    def _build_model(self) -> nn.Module:
        # initialize backbones
        if self.model_version not in self.AVAILABLE_MODEL_VERSIONS:
            msg = f"Model version {self.model_version} is not supported."
            raise ValueError(msg)

        backbone = MSCAN(version=self.model_version)
        decode_head = LightHamHead(version=self.model_version, num_classes=self.num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]
        return BaseSegmModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
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

        return {
            "inputs": labeled_inputs.images,
            "unlabeled_weak_images": w_u_images,
            "unlabeled_strong_images": s_u_images,
            "global_step": self.trainer.global_step,
            "steps_per_epoch": self.trainer.num_training_batches,
            "img_metas": labeled_inputs.imgs_info,
            "unlabeled_img_metas": unlabeled_img_metas,
            "masks": labeled_inputs.masks,
            "mode": "loss",
        }

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
        return MeanTeacher(base_model, unsup_weight=0.7, drop_unrel_pixels_percent=20, semisl_start_epoch=2)
