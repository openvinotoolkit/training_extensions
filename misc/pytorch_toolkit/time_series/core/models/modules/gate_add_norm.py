import torch
import torch.nn as nn
from .gated_linear_unit import GatedLinearUnit

class GateAddNorm(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            dropout
    ):
        super().__init__()
        self.glu = GatedLinearUnit(input_size, output_size, dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, skip):
        return self.norm(self.glu(x) + skip)
