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
import torch.nn as nn
from .embedding import *


class LengthTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(1., dtype=torch.float))

    def forward(self, z, z_mask, ls, max_len=None):
        n = z_mask.sum(1)
        max_len = ls.max().long().item() if max_len is None else max_len
        arange_l = torch.arange(max_len).to(z.device)
        arange_z = torch.arange(z.size(1)).to(z.device)

        arange_l = arange_l[None, :].repeat(z.size(0), 1).float()
        mu = arange_l * n[:, None].float() / ls[:, None].float()
        arange_z = arange_z[None, None, :].repeat(z.size(0), max_len, 1).float()
        distance = torch.clamp(arange_z - mu[:, :, None], -100, 100)
        logits = - torch.pow(2, distance) / (2. * self.sigma ** 2)

        logits = logits.float() * z_mask[:, None, :].float() - float(99) * (float(1) - z_mask[:, None, :].float())
        weight = torch.softmax(logits, 2)
        z_prime = (z[:, None, :, :] * weight[:, :, :, None]).sum(2)
        z_prime_mask = (arange_l < ls[:, None].float()).float()

        z_prime = z_prime * z_prime_mask[:, :, None]
        return z_prime, z_prime_mask


class LengthConverter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_embed_layer = PositionalEmbedding(self.hidden_size)
        self.length_embed_layer = nn.Embedding(500, self.hidden_size)
        self.length_transformer = LengthTransformer()

    def forward(self, src, src_mask, tgt_lens, max_len=None):
        rc = 1. / (2 ** 0.5)
        converted_vectors, _ = self.length_transformer(src, src_mask, tgt_lens, max_len)
        pos_embed = self.pos_embed_layer(converted_vectors)
        len_embed = self.length_embed_layer(tgt_lens.long())
        converted_vectors = rc * converted_vectors + 0.5 * pos_embed + 0.5 * len_embed[:, None, :]
        return converted_vectors
