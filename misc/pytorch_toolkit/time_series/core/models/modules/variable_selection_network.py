import torch
import torch.nn as nn
import torch.nn.functional as F
from .gated_residual_network import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    def __init__(
            self,
            hidden_size,
            output_size,
            dropout=0,
            input_size=None,
            context_size=None
    ):
        super().__init__()
        self.grn_vx = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            context_size=context_size
        )
        self.grn_feature = nn.ModuleList()
        for i in range(output_size):
            self.grn_feature.append(
                GatedResidualNetwork(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout
                )
            )
        self.size_flatten = hidden_size * output_size

    def forward(self, emb, context=None):
        if context is not None:
            context = context.unsqueeze(1)
            flatten = torch.flatten(emb, start_dim=2)
            weights = F.softmax(self.grn_vx(flatten, context), dim=-1).unsqueeze(2)
            features = torch.stack(
                [self.grn_feature[i](emb[Ellipsis, i]) for i in range(len(self.grn_feature))],
                axis=-1
            )
            outputs = torch.sum(weights * features, dim=-1)
        else:
            flatten = torch.flatten(emb, start_dim=1)
            weights = F.softmax(self.grn_vx(flatten), dim=-1).unsqueeze(-1)
            features = torch.cat(
                [self.grn_feature[i](emb[Ellipsis, i:i + 1, :]) for i in range(len(self.grn_feature))],
                axis=1
            )
            outputs = torch.sum(weights * features, dim=1)
        return outputs, weights
