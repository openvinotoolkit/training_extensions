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


class LocalPushLoss(nn.Module):
    def __init__(self, margin=0.1, weight=1.0, smart_margin=True):
        super(LocalPushLoss, self).__init__()
        self.margin = margin
        assert self.margin >= 0.0
        self.weight = weight
        self.smart_margin = smart_margin

    def forward(self, normalized_embeddings, cos_theta, target):
        if self.weight == 0.0:
            return torch.FloatTensor([0.0]).mean()
        similarity = normalized_embeddings.matmul(normalized_embeddings.permute(1, 0))

        with torch.no_grad():
            pairs_mask = target.view(-1, 1) != target.view(1, -1)

            if self.smart_margin:
                center_similarity = cos_theta[torch.arange(cos_theta.size(0), device=target.device), target]
                threshold = center_similarity.clamp(min=self.margin).view(-1, 1) - self.margin
            else:
                threshold = self.margin
            similarity_mask = similarity > threshold

            mask = pairs_mask & similarity_mask

        filtered_similarity = torch.where(mask, similarity - threshold, torch.zeros_like(similarity))
        losses, _ = filtered_similarity.max(dim=-1)

        return self.weight * losses.mean()
