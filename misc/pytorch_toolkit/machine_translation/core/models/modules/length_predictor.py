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
import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthPredictionLoss(nn.Module):
    def __init__(self, max_delta=50):
        super().__init__()
        self.max_delta = max_delta

    def forward(self, logits, src_mask, tgt_mask):
        src_lens, tgt_lens = src_mask.sum(1), tgt_mask.sum(1)
        delta = (tgt_lens - src_lens + self.max_delta).clamp(0, self.max_delta * 2 - 1).long()
        loss = F.cross_entropy(logits, delta, reduction="mean")
        return {"length_prediction_loss": loss}


class LengthPredictor(nn.Module):
    def __init__(self, hidden_size, max_delta=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_delta = max_delta
        self._init_modules()
        self._init_loss()

    def forward(self, src, src_mask, tgt_len=None):
        src_mean = self._compute_mean_emb(src, src_mask)
        logits, delta = self._predict_delta(src_mean)
        return logits, delta

    def _predict_delta(self, src):
        logits = self.length_predictor(src)
        delta = logits.argmax(-1) - float(self.max_delta)
        return logits, delta

    def _compute_mean_emb(self, src, src_mask):
        mean_emb = (src * src_mask[:, :, None]).sum(1) / src_mask.sum(1)[:, None]
        return mean_emb

    def _init_modules(self):
        self.length_predictor = nn.Linear(self.hidden_size, self.max_delta * 2)

    def _init_loss(self):
        self.loss = LengthPredictionLoss(self.max_delta)
