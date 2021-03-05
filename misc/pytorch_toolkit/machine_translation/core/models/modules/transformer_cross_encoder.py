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
from .multi_head_attention import *
from .transformer_encoder import TransformerFeedForward


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1):
        super().__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio)
        self.cross_attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)
        self.layer_norm3 = nn.LayerNorm(size)

    def forward(self, x, x_mask, y, y_mask):
        # Attention layer
        h1 = self.layer_norm1(x)
        h1, _ = self.attention(h1, h1, h1, mask=x_mask)
        h1 = self.dropout(h1)
        h1 += x
        # Cross-attention
        h2 = self.layer_norm2(h1)
        h2, _ = self.attention(h2, y, y, mask=y_mask)
        h2 = self.dropout(h2)
        h2 += h1
        # Feed-forward layer
        h3 = self.layer_norm3(h2)
        h3 = self.ff_layer(h3)
        h3 = self.dropout(h3)
        h3 += h2
        return h3


class TransformerCrossEncoder(nn.Module):
    """
    Self-attention -> cross-attenion -> FF -> layer norm
    """
    def __init__(self, embed_layer, size, n_layers, ff_size=None, n_att_head=8, dropout_ratio=0.1, skip_connect=False):
        super().__init__()
        if ff_size is None:
            ff_size = size * 4
        self._skip = skip_connect
        self._reslace = 1. / math.sqrt(2)
        self.embed_layer = embed_layer
        self.layer_norm = nn.LayerNorm(size)
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerCrossEncoderLayer(size, ff_size, n_att_head=n_att_head,
                                            dropout_ratio=dropout_ratio)
            self.encoder_layers.append(layer)

    def forward(self, x, x_mask, y, y_mask):
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        first_x = x
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, x_mask, y, y_mask)
            if self._skip:
                x = self._reslace * (first_x + x)
        x = self.layer_norm(x)
        return x
