"""MLP block module."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

from __future__ import annotations

from torch import Tensor, nn


class MLPBlock(nn.Module):
    """MLPBlock module.

    Reference: https://github.com/facebookresearch/segment-anything

    Args:
        embedding_dim (int): Embedding dimension.
        mlp_dim (int): MLP dimension.
        act (Type[nn.Module], optional): Activation function. Defaults to nn.GELU.
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.GELU,
    ) -> None:
        """Initializes an MLPBlock module.

        Args:
            embedding_dim (int): The input embedding dimension.
            mlp_dim (int): The dimension of the intermediate MLP layer.
            act (type[nn.Module], optional): The activation function to use. Defaults to nn.GELU.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLPBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.lin2(self.act(self.lin1(x)))
