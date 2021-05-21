import torch
import torch.nn as nn
import torch.nn.functional as F
from .gated_linear_unit import GatedLinearUnit


class GatedResidualNetwork(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            context_size=None,
            dropout=0
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(input_size, hidden_size)
        self.w3 = None if context_size is None else nn.Linear(context_size, hidden_size, bias=False)
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.residual = nn.Sequential() if input_size == output_size else nn.Linear(input_size, output_size)

    def forward(self, a, c=None):
        if c is not None:
            n2 = F.elu(self.w2(a) + self.w3(c))
        else:
            n2 = F.elu(self.w2(a))
        n1 = self.w1(n2)
        grn = self.layer_norm(self.residual(a) + self.glu(n1))
        return grn
