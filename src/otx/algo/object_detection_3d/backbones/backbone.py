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
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, num_pos_feats: int = 256):
        """Positional embedding."""
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward pass of the PositionEmbeddingLearned module.

        Args:
            tensor_list (NestedTensor): Input tensor.

        Returns:
            torch.Tensor: Position embeddings.
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) / w * 49
        j = torch.arange(h, device=x.device) / h * 49
        x_emb = self.get_embed(i, self.col_embed)
        y_emb = self.get_embed(j, self.row_embed)
        return (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )

    def get_embed(self, coord: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
        """Get the embedding for the given coordinates.

        Args:
            coord (torch.Tensor): The coordinates.
            embed (nn.Embedding): The embedding layer.

        Returns:
            torch.Tensor: The embedding for the coordinates.
        """
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=49)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta


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
    N_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


class BackboneBase(nn.Module):
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
        xs = self[0](images)
        out: list[NestedTensor] = []
        pos: list[torch.Tensor] = []
        for _, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


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
        backbone = Backbone(**cls.CFG[model_name])
        position_embedding = build_position_encoding(**cls.CFG[model_name]["positional_encoding"])
        return Joiner(backbone, position_embedding)
