import torch
import torch.nn as nn
from .interpretable_multi_head_attention import InterpretableMultiHeadAttention
from .gate_add_norm import GateAddNorm


class TemporalSelfAttention(nn.Module):
    def __init__(self, input_size, num_heads, dropout):
        super().__init__()
        self.self_attn = InterpretableMultiHeadAttention(
            n_head = num_heads,
            d_model = input_size,
            dropout = dropout
        )
        self.gate_add_norm = GateAddNorm(
            input_size,
            input_size,
            dropout
        )

    def forward(self, x):
        out, _ = self.self_attn(x, x, x, self._get_attn_mask(x))
        out = self.gate_add_norm(out, x)
        return out

    def _get_attn_mask(self, var):
        b, l = var.shape[:2]
        eye = torch.eye(l).to(var.device)
        mask = torch.cumsum(eye, 0).repeat(b, 1, 1).to(torch.float32).data
        return mask
