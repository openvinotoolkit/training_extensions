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
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_ratio=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, query, keys, values, mask=None):
        attn = torch.matmul(query, keys.transpose(-2, -1))
        attn /= math.sqrt(query.shape[-1])
        # apply mask
        if mask is None:
            mask = attn.new_ones(attn.shape)
        if mask.dim() < attn.dim():
            mask = mask.unsqueeze(-2)
        mask = self.dropout(mask)
        attn = attn.masked_fill(mask == 0, -1e3)
        # compute attn & mask
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, values)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, out_size, num_head=8, hidden_size=None, dropout_ratio=0):
        super().__init__()
        if hidden_size is None:
            hidden_size = out_size
        self.num_head = num_head
        self.hidden_size = hidden_size

        self.attention = ScaledDotProductAttention(dropout_ratio=dropout_ratio)
        self.proj = nn.Linear(hidden_size, out_size)

        self.query = nn.Linear(out_size, hidden_size)
        self.key = nn.Linear(out_size, hidden_size)
        self.values = nn.Linear(out_size, hidden_size)

    def forward(self, query, keys, values, mask=None):
        batch_size = query.shape[0]
        head_dim = self.hidden_size // self.num_head
        query = self.query(query).view(batch_size, -1, self.num_head, head_dim).transpose(1, 2)
        keys = self.key(keys).view(keys.shape[0], -1, self.num_head, head_dim).transpose(1, 2)
        values = self.values(values).view(values.shape[0], -1, self.num_head, head_dim).transpose(1, 2)
        if mask is not None and mask.dim() < keys.dim():
            mask = mask.unsqueeze(1)
        output, attn = self.attention(query, keys, values, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * head_dim)  # (B, T2, H)
        output = self.proj(output)
        return output, attn
