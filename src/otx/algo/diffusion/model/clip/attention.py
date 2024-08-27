"""This module provides the ClipAttention class."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class Attention(nn.Module):
    """Attention module for CLIP."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 768
        self.num_heads = 12
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, causal_attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ClipAttention module.

        Args:
            hidden_states (torch.Tensor): The input tensor.
            causal_attention_mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        bsz, tgt_len, embed_dim = hidden_states.shape
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k, v = (x.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) for x in (q, k, v))
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_attention_mask)
        return self.out_proj(attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim))
