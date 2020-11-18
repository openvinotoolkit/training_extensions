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

class LabelSmoothingKLDivLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        self.padding_idx = ignore_index
        super(LabelSmoothingKLDivLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction="sum")


class TokenPredictionLoss(nn.Module):
    def __init__(self, tgt_vocab_size, ignore_index=-100, ignore_first_token=False):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index
        self.ignore_first_token = ignore_first_token
        self.label_smooth = LabelSmoothingKLDivLoss(0.1, self.tgt_vocab_size, self.ignore_index)

    def forward(self, logits, tgt, tgt_mask, denominator=None):
        B, T, _ = logits.shape
        logits = F.log_softmax(logits, dim=2)
        flat_logits = logits.contiguous().view(B * T, self.tgt_vocab_size)
        if self.ignore_first_token:
            tgt = tgt[:, 1:]
            tgt_mask = tgt_mask[:, 1:]
        flat_targets = tgt.contiguous().view(B * T)
        flat_mask = tgt_mask.contiguous().view(B * T)
        denominator = tgt.shape[0] if denominator is None else denominator
        out = {}
        out["token_prediction_loss"] = self.label_smooth(flat_logits, flat_targets) / denominator
        out["per_word_accuracy"] = self.compute_word_accuracy(logits, tgt, tgt_mask, denominator, False) * float(tgt_mask.shape[0]) / tgt_mask.float().sum()
        return out

    def compute_word_accuracy(self, logits, tgt_seq, tgt_mask, denominator=None, ignore_first_token=True):
        """Compute per-word accuracy."""
        preds = logits.argmax(2)
        if ignore_first_token:
            tgt_seq = tgt_seq[:, 1:]
            tgt_mask = tgt_mask[:, 1:]
        if denominator is None:
            denominator = (tgt_mask.sum()).float()
        word_acc = (preds.eq(tgt_seq).float() * tgt_mask).sum() / denominator
        return word_acc
