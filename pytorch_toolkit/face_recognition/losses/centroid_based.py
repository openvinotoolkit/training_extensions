"""
 Copyright (c) 2018 Intel Corporation
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
import torch
import torch.nn.functional as F
import numpy as np


class CenterLoss(nn.Module):
    """Implements the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""
    def __init__(self, num_classes, embed_size, cos_dist=True):
        super().__init__()
        self.cos_dist = cos_dist
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, embed_size).cuda())
        self.embed_size = embed_size
        self.mse = nn.MSELoss(reduction='elementwise_mean')

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def forward(self, features, labels):
        features = F.normalize(features)
        batch_size = labels.size(0)
        features_dim = features.size(1)
        assert features_dim == self.embed_size

        if self.cos_dist:
            self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

        centers_batch = self.centers[labels, :]

        if self.cos_dist:
            cos_sim = nn.CosineSimilarity()
            cos_diff = 1. - cos_sim(features, centers_batch)
            center_loss = torch.sum(cos_diff) / batch_size
        else:
            center_loss = self.mse(centers_batch, features)

        return center_loss


class MinimumMargin(nn.Module):
    """Implements the Minimum margin loss from https://arxiv.org/abs/1805.06741"""
    def __init__(self, margin=.6):
        super().__init__()
        self.margin = margin

    def forward(self, centers, labels):
        loss_value = 0

        batch_centers = centers[labels, :]
        labels = labels.cpu().data.numpy()

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])
        valid_pairs = (all_pairs * np.tri(*all_pairs.shape, k=-1, dtype=np.bool)).astype(np.float32)
        losses = 1. - torch.mm(batch_centers, torch.t(batch_centers)) - self.margin

        valid_pairs *= (losses.cpu().data.numpy() > 0.0)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class GlobalPushPlus(nn.Module):
    """Implements the Global Push Plus loss"""
    def __init__(self, margin=.6):
        super().__init__()
        self.min_margin = 0.15
        self.max_margin = margin
        self.num_calls = 0

    def forward(self, features, centers, labels):
        self.num_calls += 1
        features = F.normalize(features)
        loss_value = 0
        batch_centers = centers[labels, :]
        labels = labels.cpu().data.numpy()
        assert len(labels.shape) == 1

        center_ids = np.arange(centers.shape[0], dtype=np.int32)
        different_class_pairs = labels.reshape([-1, 1]) != center_ids.reshape([1, -1])

        pos_distances = 1.0 - torch.sum(features * batch_centers, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(centers))

        margin = self.min_margin + float(self.num_calls) / float(40000) * (self.max_margin - self.min_margin)
        margin = min(margin, self.max_margin)

        losses = margin + pos_distances.view(-1, 1) - neg_distances

        valid_pairs = (different_class_pairs * (losses.cpu().data.numpy() > 0.0)).astype(np.float32)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class PushPlusLoss(nn.Module):
    """Implements the Push Plus loss"""
    def __init__(self, margin=.7):
        super().__init__()
        self.margin = margin

    def forward(self, features, centers, labels):
        features = F.normalize(features)
        loss_value = 0
        batch_centers = centers[labels, :]
        labels = labels.cpu().data.numpy()
        assert len(labels.shape) == 1

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])
        pos_distances = 1.0 - torch.sum(features * batch_centers, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(features))

        losses = self.margin + pos_distances.view(-1, 1) - neg_distances
        valid_pairs = (all_pairs * (losses.cpu().data.numpy() > 0.0)).astype(np.float32)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class PushLoss(nn.Module):
    """Implements the Push loss"""
    def __init__(self, soft=True, margin=0.5):
        super().__init__()
        self.soft = soft
        self.margin = margin

    def forward(self, features, labels):
        features = F.normalize(features)
        loss_value = 0
        labels = labels.cpu().data.numpy()
        assert len(labels.shape) == 1

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])
        valid_pairs = (all_pairs * np.tri(*all_pairs.shape, k=-1, dtype=np.bool)).astype(np.float32)

        if self.soft:
            losses = torch.log(1. + torch.exp(torch.mm(features, torch.t(features)) - 1))
            num_valid = float(np.sum(valid_pairs))
        else:
            losses = self.margin - (1. - torch.mm(features, torch.t(features)))
            valid_pairs *= (losses.cpu().data.numpy() > 0.0)
            num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid
