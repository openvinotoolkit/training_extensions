import torch
import torch.nn as nn
from .modules import *


class TemporalFusionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._init_model()

    def forward(self, inputs):
        regular_inputs = inputs[:, :, :self.cfg.num_regular_variables].to(torch.float)
        categorical_inputs = inputs[:, :, self.cfg.num_regular_variables:].to(torch.long)
        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.embedding(
            regular_inputs, categorical_inputs
        )
        past_inputs = self.cat_past_inputs(
            unknown_inputs,
            known_combined_layer,
            obs_inputs
        )
        future_inputs = known_combined_layer[:, self.cfg.num_encoder_steps:, :]
        # Encoder
        c_s, c_e, c_h, c_c = self.static_covariate_encoders(static_inputs)
        past_features, _ = self.past_vsn(past_inputs, c_s)
        past_lstm, (state_h, state_c) = self.past_lstm(
            past_features, (c_h.unsqueeze(0), c_c.unsqueeze(0))
        )
        future_features, _ = self.future_vsn(future_inputs, c_s)
        future_lstm, _ = self.future_lstm(future_features, (state_h, state_c))
        # Apply gated skip connection
        temporal_features = self.gate_add_norm(
            torch.cat((past_lstm, future_lstm), axis=1),
            torch.cat((past_features, future_features), axis=1)
        )
        # Decoder
        static_enrichment = self.static_enrichment(temporal_features, c_e.unsqueeze(1))
        temporal_self_attention = self.temporal_self_attention(static_enrichment)
        decoder = self.position_wise_feed_forward(temporal_self_attention, temporal_features)
        # Predictor
        output = self.predictor(decoder[:, self.cfg.num_encoder_steps:])
        return output

    def cat_past_inputs(self, unknown_inputs, known_combined_layer, obs_inputs):
        past_inputs = [
            known_combined_layer[:, :self.cfg.num_encoder_steps, :],
            obs_inputs[:, :self.cfg.num_encoder_steps, :]
        ]
        if unknown_inputs is not None:
            past_inputs = [unknown_inputs[:, :self.cfg.num_encoder_steps, :]] + past_inputs
        return torch.cat(past_inputs, axis=-1)

    def _init_model(self):
        self.embedding = TFTEmbedding(
            self.cfg.num_categorical_variables,
            self.cfg.category_counts,
            self.cfg.known_categorical_input_idx,
            self.cfg.num_regular_variables,
            self.cfg.known_regular_input_idx,
            self.cfg.static_input_idx,
            self.cfg.input_obs_idx,
            self.cfg.hidden_size
        )
        # Encoder
        self.static_covariate_encoders = StaticCovariateEncoders(
            static_input_size=len(self.cfg.static_input_idx),
            hidden_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout
        )
        self.past_vsn = VariableSelectionNetwork(
            input_size=self.cfg.hidden_size * self.embedding.get_num_non_static_past_inputs(),
            hidden_size=self.cfg.hidden_size,
            output_size=self.embedding.get_num_non_static_past_inputs(),
            context_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout
        )
        self.past_lstm = nn.LSTM(
            input_size = self.cfg.hidden_size,
            hidden_size = self.cfg.hidden_size,
            batch_first = True
        )
        self.future_vsn = VariableSelectionNetwork(
            input_size=self.cfg.hidden_size * self.embedding.get_num_non_static_future_inputs(),
            hidden_size=self.cfg.hidden_size,
            output_size=self.embedding.get_num_non_static_future_inputs(),
            context_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout
        )
        self.future_lstm = nn.LSTM(
            input_size = self.cfg.hidden_size,
            hidden_size = self.cfg.hidden_size,
            batch_first = True
        )
        self.gate_add_norm = GateAddNorm(
            self.cfg.hidden_size,
            self.cfg.hidden_size,
            self.cfg.dropout
        )
        # Decoder
        self.static_enrichment = GatedResidualNetwork(
            input_size = self.cfg.hidden_size,
            hidden_size = self.cfg.hidden_size,
            output_size = self.cfg.hidden_size,
            context_size = self.cfg.hidden_size,
            dropout = self.cfg.dropout
        )
        self.temporal_self_attention = TemporalSelfAttention(
            input_size=self.cfg.hidden_size,
            num_heads=self.cfg.num_heads,
            dropout=self.cfg.dropout
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            input_size=self.cfg.hidden_size,
            output_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout
        )
        # Predictor
        self.predictor = torch.nn.Linear(
            self.cfg.hidden_size,
            self.cfg.output_size * len(self.cfg.quantiles)
        )
