"""This module contains the definition of the ClipEncoderLayer class, which is an encoder layer for CLIP."""

import torch
from torch import nn

from .attention import Attention
from .mlp import MLP


class EncoderLayer(nn.Module):
    """Encoder layer for CLIP."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.layer_norm1 = nn.LayerNorm(768)
        self.mlp = MLP()
        self.layer_norm2 = nn.LayerNorm(768)

    def forward(self, hidden_states: torch.Tensor, causal_attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor containing hidden states.
            causal_attention_mask (torch.Tensor): Mask tensor for causal attention.

        Returns:
            torch.Tensor: Output tensor after applying self-attention and MLP.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
