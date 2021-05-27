import torch
import torch.nn as nn

class GatedLinearUnit(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            dropout=0
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w4 = nn.Linear(input_size, output_size)
        self.w5 = nn.Linear(input_size, output_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.act(self.w4(x)) * self.w5(x)
        return x
