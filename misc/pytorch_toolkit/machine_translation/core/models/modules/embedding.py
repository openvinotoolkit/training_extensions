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
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, size, max_len=5000):
        super().__init__()
        self.size = size
        self.max_len = max_len
        self.register_buffer("pe", self.build_positional_embedding(self.size, self.max_len))

    def forward(self, x, precompute=True):
        if not precompute:
            return self.build_positional_embedding(self.size, x.shape[1]).to(x.device)
        return self.pe[:, :int(x.shape[1])]

    @staticmethod
    def build_positional_embedding(size, max_len):
        emb = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, size, 2).float() *
                              -(math.log(10000.0) / size)).float())
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        emb = emb.unsqueeze(0)
        return emb


class TransformerEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, dropout_ratio=0.1):
        super().__init__(num_embeddings, embedding_dim)
        self.pos_layer = PositionalEmbedding(embedding_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, positional_encoding=True):
        embed = super().forward(x)
        embed = embed * math.sqrt(self.embedding_dim)
        if positional_encoding:
            if embed.dim() == 2:
                pos_embed = self.pos_layer(embed.unsqueeze(1), start=start)
                pos_embed = pos_embed.squeeze(1)
            else:
                pos_embed = self.pos_layer(embed)
            embed += pos_embed
        return self.dropout(embed)
