import torch
import torch.nn as nn


class TFTEmbedding(nn.Module):
    def __init__(
            self,
            num_categorical_variables,
            category_counts,
            known_categorical_input_idx,
            num_regular_variables,
            known_regular_input_idx,
            static_input_idx,
            input_obs_idx,
            hidden_size
    ):
        super().__init__()
        self.num_categorical_variables = num_categorical_variables
        self.category_counts = category_counts
        self.known_categorical_input_idx = known_categorical_input_idx
        self.num_regular_variables = num_regular_variables
        self.known_regular_input_idx = known_regular_input_idx
        self.static_input_idx = static_input_idx
        self.input_obs_idx = input_obs_idx
        self.hidden_size = hidden_size

        self.categorical_var_embeddings = nn.ModuleList([
            nn.Embedding(self.category_counts[i], self.hidden_size) for i in range(self.num_categorical_variables)
        ])
        self.regular_var_embeddings = nn.ModuleList([
            nn.Linear(1, self.hidden_size) for i in range(self.num_regular_variables)
        ])

    def forward(self, regular_inputs, categorical_inputs):
        unknown_inputs = self.get_unknown_inputs(regular_inputs, categorical_inputs)
        known_combined_layer = self.get_known_inputs(regular_inputs, categorical_inputs)
        obs_inputs = self.get_obs_inputs(regular_inputs)
        static_inputs = self.get_static_inputs(regular_inputs, categorical_inputs)
        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def get_static_inputs(self, regular_inputs, categorical_inputs):
        if not self.static_input_idx:
            return None

        static_regular_inputs = []
        for i in range(self.num_regular_variables):
            if i in self.static_input_idx:
                static_regular_inputs.append(
                    self.regular_var_embeddings[i](regular_inputs[:, 0, i:i + 1])
                )

        static_categorical_inputs = []
        for i in range(self.num_categorical_variables):
            if i + self.num_regular_variables in self.static_input_idx:
                static_categorical_inputs.append(
                    self.categorical_var_embeddings[i](categorical_inputs[Ellipsis, i])[:, 0, :]
                )

        static_inputs = torch.stack(static_regular_inputs + static_categorical_inputs, axis = 1)
        return static_inputs

    def get_obs_inputs(self, regular_inputs):
        obs_inputs = []
        for i in self.input_obs_idx:
            obs_inputs.append(
                self.regular_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
            )
        obs_inputs = torch.stack(obs_inputs, axis=-1)
        return obs_inputs

    def get_unknown_inputs(self, regular_inputs, categorical_inputs):
        wired_embeddings = []
        for i in range(self.num_categorical_variables):
            if i not in self.known_categorical_input_idx and i not in self.input_obs_idx:
                e = self.categorical_var_embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(self.num_regular_variables):
            if i not in self.known_regular_input_idx and i not in self.input_obs_idx:
                e = self.regular_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None

        return unknown_inputs

    def get_known_inputs(self, regular_inputs, categorical_inputs):
        known_regular_inputs = []
        for i in self.known_regular_input_idx:
            if i not in self.static_input_idx:
                known_regular_inputs.append(
                    self.regular_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                )

        known_categorical_inputs = []
        for i in self.known_categorical_input_idx:
            if i + self.num_regular_variables not in self.static_input_idx:
                known_categorical_inputs.append(
                    self.categorical_var_embeddings[i](categorical_inputs[Ellipsis, i])
                )

        known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, axis=-1)
        return known_combined_layer

    def get_num_non_static_future_inputs(self):
        num_known_regular_inputs = len(
            [i for i in self.known_regular_input_idx if i not in self.static_input_idx]
        )
        num_known_categorical_inputs = len(
            [i for i in self.known_categorical_input_idx if i + self.num_regular_variables not in self.static_input_idx]
        )
        return num_known_regular_inputs + num_known_categorical_inputs

    def get_num_unknown_inputs(self):
        wired_embeddings = len(
            [i for i in range(self.num_categorical_variables) if i not in self.known_categorical_input_idx and i not in self.input_obs_idx]
        )
        unknown_inputs = len(
            [i for i in range(self.num_regular_variables) if i not in self.known_regular_input_idx and i not in self.input_obs_idx]
        )
        return wired_embeddings + unknown_inputs

    def get_num_non_static_past_inputs(self):
        return len(self.input_obs_idx) + self.get_num_unknown_inputs() + self.get_num_non_static_future_inputs()
