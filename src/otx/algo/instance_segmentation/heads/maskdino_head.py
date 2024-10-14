# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskDINO Head module."""

from __future__ import annotations

from torch import Tensor, nn

from otx.algo.instance_segmentation.heads.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from otx.algo.instance_segmentation.heads.transformer_decoder.maskdino_decoder import MaskDINODecoder


class MaskDINOHead(nn.Module):
    """MaskDINO Head module.

    Args:
        num_classes (int): number of classes
        pixel_decoder (MaskDINOEncoder): pixel decoder
        transformer_predictor (MaskDINODecoder): mask transformer predictor
    """

    def __init__(
        self,
        num_classes: int,
        pixel_decoder: MaskDINOEncoder,
        transformer_predictor: MaskDINODecoder,
    ):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.num_classes = num_classes

    def forward(
        self,
        features: dict[str, Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Forward pass."""
        mask_features, _, multi_scale_features = self.pixel_decoder(features)
        return self.predictor(multi_scale_features, mask_features, targets=targets)
