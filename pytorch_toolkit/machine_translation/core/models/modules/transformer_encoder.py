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
import math
import torch.nn as nn
import torch.nn.functional as F
from .multi_head_attention import *


class TransformerFeedForward(nn.Module):
    def __init__(self, size, hidden_size, dropout_ratio=0.1, act="relu"):
        super().__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.act = getattr(F, act)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1):
        super().__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)

    def forward(self, x, src_mask=None):
        # Attention layer
        y1 = self.layer_norm1(x)
        y1, _ = self.attention(y1, y1, y1, mask=src_mask)
        y1 = self.dropout(y1)
        y1 += x
        # Feed-forward layer
        y2 = self.layer_norm2(y1)
        y2 = self.ff_layer(y2)
        y2 = self.dropout(y2)
        y2 += y1
        return y2


class TransformerEncoder(nn.Module):
    def __init__(self, embed_layer, size, n_layers, ff_size=None, n_att_head=8, dropout_ratio=0.1, skip_connect=False):
        super().__init__()
        if ff_size is None:
            ff_size = size * 4
        self.embed_layer = embed_layer
        self.layer_norm = nn.LayerNorm(size)
        self.layers = nn.ModuleList()
        self.skip_connect = skip_connect
        self.scale = 1. / math.sqrt(2)
        for _ in range(n_layers):
            self.layers.append(
                TransformerEncoderLayer(size, ff_size, n_att_head, dropout_ratio)
            )

    def forward(self, x, mask=None):
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        first_x = x
        for l, layer in enumerate(self.layers):
            x = layer(x, mask)
            if self.skip_connect:
                x = self.scale * (first_x + x)
        x = self.layer_norm(x)
        return x
