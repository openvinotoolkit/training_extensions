# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MonoDetr backbone implementations."""
from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from otx.algo.modules.norm import FrozenBatchNorm2d
from otx.algo.object_detection_3d.utils.utils import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        """Initialize the PositionEmbeddingSine module.

        Args:
            num_pos_feats (int): Number of positional features.
            temperature (int): Temperature scaling factor.
            normalize (bool): Flag indicating whether to normalize the position embeddings.
            scale (Optional[float]): Scaling factor for the position embeddings. If None, default value is used.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            msg = "normalize should be True if scale is passed"
            raise ValueError(msg)
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward function for PositionEmbeddingSine module."""
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


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
        position_embedding: PositionEmbeddingSine,
    ) -> None:
        """Initialize the Joiner module.

        Args:
            backbone (nn.Module): The backbone module.
            position_embedding (Union[PositionEmbeddingSine]): The position embedding module.
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
            },
        },
    }

    def __new__(cls, model_name: str) -> Joiner:
        """Constructor for Backbone MonoDetr."""
        # TODO (Kirill): change backbone to already implemented in OTX
        backbone = Backbone(**cls.CFG[model_name])
        n_steps = cls.CFG[model_name]["positional_encoding"]["hidden_dim"] // 2
        position_embedding = PositionEmbeddingSine(n_steps, normalize=True)
        return Joiner(backbone, position_embedding)
