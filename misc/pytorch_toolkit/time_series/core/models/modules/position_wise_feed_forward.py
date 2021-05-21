import torch
import torch.nn as nn
from .gated_residual_network import GatedResidualNetwork
from .gate_add_norm import GateAddNorm


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=input_size,
            output_size=output_size,
            dropout=dropout
        )
        self.gate_add_norm = GateAddNorm(
            input_size=input_size,
            output_size=output_size,
            dropout=dropout
        )

    def forward(self, x, skip):
        out = self.grn(x)
        out = self.gate_add_norm(out, skip)
        return out
