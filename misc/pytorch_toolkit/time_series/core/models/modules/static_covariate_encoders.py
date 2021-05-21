import torch
import torch.nn as nn
from .gated_residual_network import GatedResidualNetwork
from .variable_selection_network import VariableSelectionNetwork


class StaticCovariateEncoders(nn.Module):
    def __init__(
            self,
            static_input_size,
            hidden_size,
            dropout
    ):
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            input_size=hidden_size * static_input_size,
            hidden_size=hidden_size,
            output_size=static_input_size,
            dropout=dropout
        )
        grn_params = {
            "input_size": hidden_size,
            "hidden_size": hidden_size,
            "output_size": hidden_size,
            "dropout": dropout
        }
        self.variable_selection_grn = GatedResidualNetwork(**grn_params)
        self.enrichment_grn = GatedResidualNetwork(**grn_params)
        self.state_h_grn = GatedResidualNetwork(**grn_params)
        self.state_c_grn = GatedResidualNetwork(**grn_params)

    def forward(self, static_inputs):
        vsn, _ = self.vsn(static_inputs)
        variable_selection = self.variable_selection_grn(vsn)
        enrichment = self.enrichment_grn(vsn)
        state_h = self.state_h_grn(vsn)
        state_c = self.state_c_grn(vsn)
        return variable_selection, enrichment, state_h, state_c
