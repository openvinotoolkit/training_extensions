# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.distributed as dist
from mmcls.models.builder import LOSSES
from torch import nn

from otx.mpa.modules.utils.distance_utils import get_dist_info


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * (
        torch.sqrt(torch.sum(torch.pow(y, 2), 1))
    ).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine


def _batch_hard(mat_distance, mat_similarity, return_indices=False):
    # TODO support distributed
    sorted_mat_distance, positive_indices = torch.sort(
        mat_distance + (-9999.0) * (1 - mat_similarity), dim=1, descending=True
    )
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(
        mat_distance + (9999.0) * (mat_similarity), dim=1, descending=False
    )
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if return_indices:
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


@LOSSES.register_module()
class TripletLoss(nn.Module):
    """
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    """

    __dist_factory = {
        "euclidean": euclidean_dist,
        "cosine": cosine_dist,
    }

    def __init__(
        self,
        margin=0.3,
        dist_metric="euclidean",
    ):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.dist_metric = self.__dist_factory[dist_metric]
        self.rank, self.world_size, self.dist = get_dist_info()

        self.margin_ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, results, targets):

        emb = results

        if self.dist:
            all_emb = [torch.empty_like(emb) for _ in range(self.world_size)]
            dist.all_gather(all_emb, emb)
            all_emb = torch.cat(all_emb).detach()

            all_targets = [torch.empty_like(targets) for _ in range(self.world_size)]
            dist.all_gather(all_targets, targets)
            all_targets = torch.cat(all_targets).detach()
        else:
            all_emb = emb.detach()
            all_targets = targets.detach()

        # mat_dist = self.dist_metric(emb, emb)
        mat_dist = self.dist_metric(emb, all_emb)
        # assert mat_dist.size(0) == mat_dist.size(1)
        N, M = mat_dist.size()
        # mat_sim = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        mat_sim = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()

        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        assert dist_an.size(0) == dist_ap.size(0)
        y = torch.ones_like(dist_ap)
        # loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        loss = self.margin_ranking_loss(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss


@LOSSES.register_module()
class SoftmaxTripletLoss(TripletLoss):
    def forward(self, results, targets):

        emb = results

        if self.dist:
            all_emb = [torch.empty_like(emb) for _ in range(self.world_size)]
            dist.all_gather(all_emb, emb)
            all_emb = torch.cat(all_emb).detach()

            all_targets = [torch.empty_like(targets) for _ in range(self.world_size)]
            dist.all_gather(all_targets, targets)
            all_targets = torch.cat(all_targets).detach()
        else:
            all_emb = emb.detach()
            all_targets = targets.detach()

        # mat_dist = self.dist_metric(emb, emb)
        # assert (mat_dist.size(0) == mat_dist.size(1)), "debug"
        mat_dist = self.dist_metric(emb, all_emb)
        N, M = mat_dist.size()
        # mat_sim = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        mat_sim = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, return_indices=True)
        assert dist_an.size(0) == dist_ap.size(0), "debug"
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = self.logsoftmax(triple_dist)

        # hard-label softmax triplet loss
        loss = (-self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
        return loss


@LOSSES.register_module()
class SoftSoftmaxTripletLoss(TripletLoss):
    def forward(self, results, targets, results_mean):
        assert results_mean is not None

        emb1 = results
        emb2 = results_mean

        if self.dist:
            all_emb1 = [torch.empty_like(emb1) for _ in range(self.world_size)]
            dist.all_gather(all_emb1, emb1)
            all_emb1 = torch.cat(all_emb1).detach()

            all_emb2 = [torch.empty_like(emb2) for _ in range(self.world_size)]
            dist.all_gather(all_emb2, emb2)
            all_emb2 = torch.cat(all_emb2).detach()

            all_targets = [torch.empty_like(targets) for _ in range(self.world_size)]
            dist.all_gather(all_targets, targets)
            all_targets = torch.cat(all_targets).detach()
        else:
            all_emb1 = emb1.detach()
            all_emb2 = emb2.detach()
            all_targets = targets.detach()

        # mat_dist = self.dist_metric(emb1, emb1)
        # assert (mat_dist.size(0) == mat_dist.size(1)), "debug"
        mat_dist = self.dist_metric(emb1, all_emb1)
        N, M = mat_dist.size()
        # mat_sim = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        mat_sim = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, return_indices=True)
        assert dist_an.size(0) == dist_ap.size(0), "debug"
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = self.logsoftmax(triple_dist)

        # mat_dist_ref = self.dist_metric(emb2, emb2)
        mat_dist_ref = self.dist_metric(emb2, all_emb2)
        # dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
        # dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, M))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, M))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = self.softmax(triple_dist_ref).detach()
        # soft-label softmax triplet loss
        loss = (-triple_dist_ref * triple_dist).mean(0).sum()
        return loss
