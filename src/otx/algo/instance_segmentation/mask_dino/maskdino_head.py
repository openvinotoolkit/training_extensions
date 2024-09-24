# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
from __future__ import annotations

from detectron2.layers import ShapeSpec
from torch import nn


class MaskDINOHead(nn.Module):
    def __init__(
        self,
        input_shape: dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
    ):
        """Args:
        input_shape: shapes (channels and stride) of the input features
        num_classes: number of classes to predict
        pixel_decoder: the pixel decoder module
        loss_weight: loss weight
        ignore_value: category id to be ignored during training.
        transformer_predictor: the transformer decoder that makes prediction
        transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes

    def forward(self, features, mask=None, targets=None):
        return self.layers(features, mask, targets=targets)

    def layers(self, features, mask=None, targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features,
            mask,
        )

        predictions = self.predictor(multi_scale_features, mask_features, mask, targets=targets)

        return predictions
