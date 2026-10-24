# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PEFT modules for Vision Transformer attention."""

import torch
from timm.models.vision_transformer import Attention


class LoRALayer(torch.nn.Module):
    """LoRA layer implementation for computing A, B composition.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        rank (int): Rank of the low-rank matrices A, B.
        alpha (float): Scaling factor applied to the output.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LoRA layer."""
        return self.alpha * (x @ self.A @ self.B)


class AttentionWithLoRA(torch.nn.Module):
    """Add LoRA layer into QKV attention layer in VisionTransformer.

    Args:
        qkv (Attention): The original QKV attention layer.
        rank (int): Rank of the low-rank matrices A, B.
        alpha (float): Scaling factor applied to the output.
    """

    def __init__(self, qkv: Attention, rank: int, alpha: float):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)
        self.weight = qkv.weight
        self.bias = qkv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AttentionWithLoRA."""
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim :] += self.lora_v(x)
        return qkv


class DoRALayer(torch.nn.Module):
    """DoRA layer implementation for Weight-Decomposed Low-Rank Adaptation.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        rank (int): Rank of the low-rank matrices A, B.
        alpha (float): Scaling factor applied to the output.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.magnitude = torch.nn.Parameter(torch.ones(out_dim))

    def forward(self, x: torch.Tensor, pretrained_weight: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DoRA layer."""
        lora_weight = self.alpha * (self.A @ self.B)
        combined_weight = pretrained_weight.T + lora_weight
        weight_norm = torch.norm(combined_weight, dim=0, keepdim=True)
        normalized_weight = combined_weight / weight_norm
        final_weight = self.magnitude.unsqueeze(0) * normalized_weight
        return x @ final_weight

    def initialize_magnitude(self, pretrained_weight: torch.Tensor) -> None:
        """Initialize the DoRA magnitude vector based on the pretrained weight."""
        with torch.no_grad():
            weight_norms = torch.norm(pretrained_weight.T, dim=0)
            self.magnitude.data = weight_norms


class AttentionWithDoRA(torch.nn.Module):
    """Add DoRA layer into QKV attention layer in VisionTransformer.

    Args:
        qkv (Attention): The original QKV attention layer.
        rank (int): Rank of the low-rank matrices A, B.
        alpha (float): Scaling factor applied to the output.
    """

    def __init__(self, qkv: Attention, rank: int, alpha: float):
        super().__init__()
        self.dim = qkv.in_features
        self.out_features = qkv.out_features

        # DoRA layers for Query and Value respectively
        self.dora_q = DoRALayer(self.dim, self.dim, rank, alpha)
        self.dora_v = DoRALayer(self.dim, self.dim, rank, alpha)

        self.weight = qkv.weight
        self.bias = qkv.bias

        self._initialize_magnitudes()

    def _initialize_magnitudes(self) -> None:
        """Initialize DoRA magnitude vector for q, v respectively."""
        q_weight = self.weight[: self.dim, :]
        v_weight = self.weight[-self.dim :, :]

        self.dora_q.initialize_magnitude(q_weight)
        self.dora_v.initialize_magnitude(v_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AttentionWithDoRA."""
        # Split the original weight into Query, Key, Value
        q_weight = self.weight[: self.dim, :]
        k_weight = self.weight[self.dim : 2 * self.dim, :]
        v_weight = self.weight[-self.dim :, :]

        # Apply DoRA to Query and Value
        q_output = self.dora_q(x, q_weight)
        k_output = x @ k_weight.T
        v_output = self.dora_v(x, v_weight)

        # Concatenate Query, Key, Value
        qkv = torch.cat([q_output, k_output, v_output], dim=-1)

        # Add bias
        if self.bias is not None:
            qkv = qkv + self.bias

        return qkv
