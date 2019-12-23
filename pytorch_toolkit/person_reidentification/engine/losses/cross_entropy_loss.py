"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

 Copyright (c) 2019 Intel Corporation

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

from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True, conf_penalty=0.):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.conf_penalty = conf_penalty

    def get_last_info(self):
        return {}

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        sm_loss = (- targets * log_probs).sum(1)

        if self.conf_penalty > 0.:
            probs = torch.exp(log_probs)
            ent = (-probs*torch.log(probs.clamp(min=1e-12))).sum(1)
            loss = nn.functional.relu(5 * sm_loss - 0.085 * ent)
            with torch.no_grad():
                nonzero_count = loss.nonzero().size(0)
            return loss.sum() / nonzero_count

        return sm_loss.mean(0)
