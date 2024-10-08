# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MonoDetr backbone implementations."""
from __future__ import annotations

from typing import Any, ClassVar

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from otx.algo.common.utils.position_embed import PositionEmbeddingLearned, PositionEmbeddingSine
from otx.algo.modules.norm import FrozenBatchNorm2d
from otx.algo.object_detection_3d.utils.utils import NestedTensor


def build_position_encoding(
    hidden_dim: int,
    position_embedding: str | PositionEmbeddingSine | PositionEmbeddingLearned,
) -> PositionEmbeddingSine | PositionEmbeddingLearned:
    """Build the position encoding module.

    Args:
        hidden_dim (int): The hidden dimension.
        position_embedding (Union[str, PositionEmbeddingSine, PositionEmbeddingLearned]): The position embedding type.

    Returns:
        Union[PositionEmbeddingSine, PositionEmbeddingLearned]: The position encoding module.
    """
    n_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(n_steps, normalize=True)
    elif position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(n_steps)
    else:
        msg = f"not supported {position_embedding}"
        raise ValueError(msg)

    return position_embedding


class BackboneBase(nn.Module):
    """BackboneBase module."""

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        """Initializes BackboneBase module."""
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images: torch.Tensor) -> dict[str, NestedTensor]:
        """Forward pass of the BackboneBase module.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            dict[str, NestedTensor]: Output tensors.
        """
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool, **kwargs):
        """Initializes Backbone module."""
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=norm_layer,
        )
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    """Joiner module."""

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: PositionEmbeddingSine | PositionEmbeddingLearned,
    ) -> None:
        """Initialize the Joiner module.

        Args:
            backbone (nn.Module): The backbone module.
            position_embedding (Union[PositionEmbeddingSine, PositionEmbeddingLearned]): The position embedding module.
        """
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images: torch.Tensor) -> tuple[list[NestedTensor], list[torch.Tensor]]:
        """Forward pass of the Joiner module.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            tuple[List[NestedTensor], List[torch.Tensor]]: Output tensors and position embeddings.
        """
        out: list[NestedTensor] = [x for _, x in sorted(self[0](images).items())]
        return out, [self[1](x).to(x.tensors.dtype) for x in out]


class BackboneBuilder:
    """DepthAwareTransformerBuilder."""

    CFG: ClassVar[dict[str, Any]] = {
        "monodetr_50": {
            "name": "resnet50",
            "train_backbone": True,
            "dilation": False,
            "return_interm_layers": True,
            "positional_encoding": {
                "hidden_dim": 256,
                "position_embedding": "sine",
            },
        },
    }

    def __new__(cls, model_name: str) -> Joiner:
        """Constructor for Backbone MonoDetr."""
        # TODO (Kirill): change backbone to already implemented in OTX
        backbone = Backbone(**cls.CFG[model_name])
        position_embedding = build_position_encoding(**cls.CFG[model_name]["positional_encoding"])
        return Joiner(backbone, position_embedding)
