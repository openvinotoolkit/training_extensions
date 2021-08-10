"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import PositionalEncodingPermute2D

FASTTEXT_EMB_DIM = 300


class Encoder(nn.Module):
    def __init__(self, dim_input, dim_internal, num_layers):
        super().__init__()

        self.dim_input = dim_input

        module_list = []
        for _ in range(num_layers):
            module_list.extend([
                nn.Conv2d(dim_input, dim_internal,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim_internal),
                nn.ReLU(inplace=True)
            ])
            dim_input = dim_internal
        self.layers = nn.Sequential(*module_list)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        feature = self.layers(feature)
        return feature


class DecoderAttention2d(nn.Module):
    str_to_class = {
        'GRU': nn.GRU,
        'LSTM': nn.LSTM
    }

    def __init__(self, hidden_size, vocab_size, decoder_input_feature_size, rnn_type):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        assert len(decoder_input_feature_size) == 2
        self.flatten_feature_size = decoder_input_feature_size[0] * \
            decoder_input_feature_size[1]

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        assert rnn_type in self.str_to_class.keys(), f'Unsupported decoder type {rnn_type}'
        self.decoder = self.str_to_class[rnn_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1, bidirectional=False)

        self.encoder_outputs_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_state_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.Tensor(self.hidden_size, 1))  # context vector

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

        nn.init.normal_(self.v, 0, 0.1)

    def forward(self, prev_symbol, hidden, encoder_outputs, cell=None):
        '''
        :param prev_symbol: Shape is [1, BATCH_SIZE]
        :param hidden: Shape is [1, BATCH_SIZE, HIDDEN_DIM]
        :param cell: Shape is [1, BATCH_SIZE, HIDDEN_DIM], it is used in case of LSTM
        :param encoder_outputs: [BATCH_SIZE, T, HIDDEN_DIM]
        :return:
        '''

        BATCH_SIZE = hidden.shape[1]
        assert tuple(hidden.shape) == (1, BATCH_SIZE, self.hidden_size), f'{hidden.shape}'
        assert tuple(prev_symbol.shape) == (BATCH_SIZE,), f'{prev_symbol.shape} {prev_symbol}'
        assert tuple(encoder_outputs.shape) == (
            BATCH_SIZE, self.flatten_feature_size, self.hidden_size), f'Got {encoder_outputs.shape} | ' \
            f'Expected batch {BATCH_SIZE}, feature size {self.flatten_feature_size}, hidden {self.hidden_size}'

        prev_symbol = prev_symbol.long()

        prev_symbol = self.embedding(prev_symbol)

        encoder_outputs_w = self.encoder_outputs_w(encoder_outputs)
        hidden_state_w = self.hidden_state_w(hidden[0]).unsqueeze(1)
        assert tuple(hidden_state_w.shape) == (BATCH_SIZE, 1, self.hidden_size)
        hidden_state_w = hidden_state_w.expand(
            (BATCH_SIZE, encoder_outputs_w.shape[1], self.hidden_size))
        assert tuple(hidden_state_w.shape) == (
            BATCH_SIZE, encoder_outputs_w.shape[1], self.hidden_size)

        s = torch.tanh(encoder_outputs_w + hidden_state_w)
        assert tuple(s.shape) == (
            BATCH_SIZE, self.flatten_feature_size, self.hidden_size)
        s = s.reshape(-1, self.hidden_size)
        s = torch.matmul(s, self.v)
        s = s.reshape(-1, self.flatten_feature_size)

        attn_weights = F.softmax(s, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        attn_applied = attn_applied.permute(1, 0, 2)
        attn_applied = attn_applied.squeeze(0)

        output = torch.cat((prev_symbol, attn_applied), 1)

        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        self.decoder.flatten_parameters()
        if isinstance(self.decoder, nn.GRU):
            output, hidden = self.decoder(output, hidden)
        elif isinstance(self.decoder, nn.LSTM):
            output, (hidden, cell) = self.decoder(output, (hidden, cell))

        hidden = torch.reshape(hidden, hidden.shape)

        output = self.out(output[0])
        output = F.log_softmax(output, dim=1)

        if isinstance(self.decoder, nn.LSTM):
            return output, hidden, cell, attn_weights
        return output, hidden, attn_weights


class TextRecognitionHeadAttention(nn.Module):

    def __init__(self,
                 decoder_vocab_size,
                 encoder_input_size,
                 encoder_dim_internal,
                 encoder_num_layers,
                 decoder_input_feature_size,
                 decoder_max_seq_len,
                 decoder_dim_hidden,
                 decoder_sos_index,
                 decoder_rnn_type,
                 dropout_ratio=0.0,
                 use_semantics=False,
                 positional_encodings=False,
                 ):
        super().__init__()

        self.encoder = Encoder(
            encoder_input_size, encoder_dim_internal, encoder_num_layers)
        self.dropout = nn.Dropout(dropout_ratio)
        self.decoder = DecoderAttention2d(hidden_size=decoder_dim_hidden,
                                          vocab_size=decoder_vocab_size,
                                          decoder_input_feature_size=decoder_input_feature_size,
                                          rnn_type=decoder_rnn_type)

        self.decoder_input_feature_size = decoder_input_feature_size
        self.decoder_max_seq_len = decoder_max_seq_len
        self.decoder_sos_int = decoder_sos_index
        self.decoder_dim_hidden = decoder_dim_hidden
        if positional_encodings:
            self.pe = PositionalEncodingPermute2D(channels=encoder_input_size)
        if use_semantics:
            dim = np.prod([decoder_dim_hidden, *decoder_input_feature_size])
            self.semantics = nn.Sequential(
                nn.Linear(dim, dim, bias=True),
                nn.ReLU(),
                nn.Linear(dim, FASTTEXT_EMB_DIM, bias=True),
            )
            self.semantic_transform = nn.Linear(FASTTEXT_EMB_DIM, decoder_dim_hidden)

    def forward(self, features, targets=None, masks=None):

        features = self.encoder(features)

        if targets is not None:
            decoder_max_seq_len = max(len(target) for target in targets)
            decoder_max_seq_len = max(decoder_max_seq_len, 1)
        else:
            decoder_max_seq_len = self.decoder_max_seq_len
        decoder_outputs = []
        batch_size = features.shape[0]
        if hasattr(self, 'pe'):
            features = features + self.pe(features)

        features = features.view(features.shape[0], features.shape[1], -1)  # B C H*W
        features = features.permute(0, 2, 1)  # BxH*WxC or BxTxC
        features = self.dropout(features)

        if hasattr(self, 'semantics'):
            assert isinstance(self.decoder.decoder, nn.GRU), "sematic module could only be applied with GRU RNN"
            old_shape = features.shape
            features = features.reshape(features.shape[0], -1)  # B C*H*W
            semantic_info = self.semantics(features)
            decoder_hidden = self.semantic_transform(semantic_info.unsqueeze(0))
            features = features.view(old_shape)
        else:
            decoder_hidden, decoder_cell = self._zero_initialize_hiddens(batch_size, features.device)
        for di in range(decoder_max_seq_len):
            if targets is not None:
                decoder_input = targets[:, di]
            else:
                decoder_input = self._extract_next_input(decoder_outputs, batch_size, features.device)
            if isinstance(self.decoder.decoder, nn.GRU):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, features)
            elif isinstance(self.decoder.decoder, nn.LSTM):
                decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(
                    decoder_input, decoder_hidden, features, decoder_cell)
            decoder_outputs.append(decoder_output)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        classes = torch.max(decoder_outputs, dim=2)[1]
        if hasattr(self, 'semantics') and self.training:
            return decoder_outputs, classes, semantic_info
        return decoder_outputs, classes

    def _extract_next_input(self, decoder_outputs, batch_size, device):
        if decoder_outputs:
            topi = torch.argmax(torch.exp(decoder_outputs[-1]), dim=1)
            decoder_input = topi.detach().view(batch_size)
        else:
            decoder_input = torch.ones([batch_size], device=device,
                                       dtype=torch.long) * self.decoder_sos_int
        return decoder_input

    def _zero_initialize_hiddens(self, batch_size, device):
        decoder_hidden = torch.zeros([1, batch_size, self.decoder_dim_hidden], device=device)
        decoder_cell = torch.zeros([1, batch_size, self.decoder_dim_hidden], device=device)
        return decoder_hidden, decoder_cell

    def dummy_forward(self):
        return torch.zeros((1, self.decoder_max_seq_len, self.decoder.vocab_size),
                           dtype=torch.float32)

    def encoder_wrapper(self, features):
        features = self.encoder(features)
        batch_size = features.shape[0]
        if hasattr(self, 'pe'):
            features = features + self.pe(features)

        features = features.view(features.shape[0], features.shape[1], -1)  # B C H*W
        features = features.permute(0, 2, 1)  # BxH*WxC or BxTxC
        features = self.dropout(features)

        if hasattr(self, 'semantics'):
            assert isinstance(self.decoder.decoder, nn.GRU), "sematic module could only be applied with GRU RNN"
            old_shape = features.shape
            features = features.reshape(features.shape[0], -1)  # B C*H*W
            semantic_info = self.semantics(features)
            decoder_hidden = self.semantic_transform(semantic_info.unsqueeze(0))
            features = features.view(old_shape)
            return features, decoder_hidden

        decoder_hidden, decoder_cell = self._zero_initialize_hiddens(batch_size, features.device)
        return features, decoder_hidden, decoder_cell

    def decoder_wrapper(self, recurrent_state, features, decoder_input):
        if isinstance(self.decoder.decoder, nn.GRU):
            hidden = recurrent_state
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, hidden, features)
            return decoder_hidden, decoder_output

        hidden, context = recurrent_state
        decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(
            decoder_input, hidden, features, context)
        return decoder_hidden, decoder_cell, decoder_output
