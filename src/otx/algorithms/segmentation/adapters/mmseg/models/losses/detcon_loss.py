"""DetCon loss."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=no-name-in-module, not-callable
import itertools

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from torch import nn


def manual_cross_entropy(logits, labels, weight):
    """Manually calculate weighted cross entropy."""
    cross_entropy = -weight * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return torch.mean(cross_entropy)


@LOSSES.register_module
class DetConLoss(nn.Module):
    """Modified from https://github.com/deepmind/detcon/blob/main/utils/losses.py.

    Compute the NCE scores from pairs of predictions and targets.
    This implements the batched form of the loss described in
    Section 3.1, Equation 3 in https://arxiv.org/pdf/2103.10957.pdf.

    Args:
        temperature: (float) the temperature to use for the NCE loss.
        use_replicator_loss (bool): use cross-replica samples.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_replicator_loss: bool = True,
        ignore_index: int = 255,
    ):  # pylint: disable=unused-argument
        super().__init__()
        assert temperature > 0
        self.temperature = torch.tensor(temperature)
        self.use_replicator_loss = use_replicator_loss

    def get_distributed_tensors(self, target1, target2, batch_size, num_samples, num_features, device):
        """Grab tensors across replicas during distributed training."""
        if dist.is_initialized() and self.use_replicator_loss:
            # Grab tensor across replicas and expand first dimension
            world_size = dist.get_world_size()
            target1_large = [torch.zeros_like(target1) for _ in range(world_size)]
            target2_large = [torch.zeros_like(target2) for _ in range(world_size)]
            dist.all_gather(target1_large, target1)
            dist.all_gather(target2_large, target2)
            target1_large = torch.cat(target1_large, dim=0)
            target2_large = torch.cat(target2_large, dim=0)

            # Fold into batch dimension
            target1_large = target1_large.reshape(-1, num_samples, num_features)
            target2_large = target2_large.reshape(-1, num_samples, num_features)

            # Create the labels by using the current replica ID and offsetting.
            replica_id = dist.get_rank()
            labels_idx = torch.arange(batch_size) + replica_id * batch_size
            enlarged_bs = target1_large.shape[0]
            labels = F.one_hot(labels_idx, num_classes=enlarged_bs).to(device)
        else:
            target1_large = target1
            target2_large = target2
            labels = F.one_hot(torch.arange(batch_size), num_classes=batch_size).to(device)

        labels = labels.unsqueeze(dim=2).unsqueeze(dim=1)

        return target1_large, target2_large, labels

    # pylint: disable=too-many-arguments, too-many-locals
    def forward(
        self,
        pred1,
        pred2,
        target1,
        target2,
        pind1,
        pind2,
        tind1,
        tind2,
        local_negatives=True,
    ):
        """Forward loss.

        Args:
            pred1 (Tensor): (b, num_samples, d) the prediction from first view.
            pred2 (Tensor): (b, num_samples, d) the prediction from second view.
            target1 (Tensor): (b, num_samples, d) the projection from first view.
            target2 (Tensor): (b, num_samples, d) the projection from second view.
            pind1 (Tensor): (b, num_samples) mask indices for first view's prediction.
            pind2 (Tensor): (b, num_samples) mask indices for second view's prediction.
            tind1 (Tensor): (b, num_samples) mask indices for first view's projection.
            tind2 (Tensor): (b, num_samples) mask indices for second view's projection.
            local_negatives (bool): whether to include local negatives.

        Returns:
            dict[str, Tensor]: A single scalar loss for the XT-NCE objective.
        """
        batch_size, num_samples, num_features = pred1.shape
        main_dtype = pred1.dtype
        # infinity_proxy is reduced to avoid overflow when training w/ fp16.
        infinity_proxy = 1e4  # Used for masks to proxy a very large number.

        def make_same_obj(ind_0, ind_1):
            same_obj = torch.eq(
                ind_0.reshape([batch_size, num_samples, 1]),
                ind_1.reshape([batch_size, 1, num_samples]),
            )
            same_obj = same_obj.unsqueeze(2).to(main_dtype)
            return same_obj

        same_obj_dict = {}
        for pair, (pind, tind) in zip(
            ["aa", "ab", "ba", "bb"],
            list(itertools.product([pind1, pind2], [tind1, tind2])),
        ):
            same_obj_dict[pair] = make_same_obj(pind, tind)

        # L2 normalize the tensors to use for the cosine-similarity
        def normalize_same_dtype(logit, p=2, dim=1, eps=1e-12, dtype=None):
            # modified from torch.nn.functional.normalize
            denom = logit.norm(p, dim, keepdim=True, dtype=dtype).clamp_min(eps).expand_as(logit)
            return logit / denom

        pred1 = normalize_same_dtype(pred1, dim=-1, dtype=main_dtype)
        pred2 = normalize_same_dtype(pred2, dim=-1, dtype=main_dtype)
        target1 = normalize_same_dtype(target1, dim=-1, dtype=main_dtype)
        target2 = normalize_same_dtype(target2, dim=-1, dtype=main_dtype)
        target1_large, target2_large, labels = self.get_distributed_tensors(
            target1, target2, batch_size, num_samples, num_features, pred1.device
        )

        # Do our matmuls and mask out appropriately.
        logits_dict = {}
        for pair, (pred, target) in zip(
            ["aa", "ab", "ba", "bb"],
            list(itertools.product([pred1, pred2], [target1_large, target2_large])),
        ):
            logits_dict[pair] = torch.einsum("abk,uvk->abuv", pred, target) / self.temperature

        labels_dict = {key: labels * same_obj for key, same_obj in same_obj_dict.items()}
        for pair in ["aa", "bb"]:
            logits_dict[pair] -= infinity_proxy * labels * same_obj_dict[pair]
            labels_dict[pair] *= 0.0

        if not local_negatives:
            for pair in ["aa", "ab", "ba", "bb"]:
                logits_dict[pair] -= infinity_proxy * labels * (1 - same_obj_dict[pair])

        labels_concat = [
            torch.cat([labels_dict["ab"], labels_dict["aa"]], dim=2).reshape((batch_size, num_samples, -1)),
            torch.cat([labels_dict["ba"], labels_dict["bb"]], dim=2).reshape((batch_size, num_samples, -1)),
        ]

        num_positives = [torch.sum(label_concat, dim=-1, keepdim=True) for label_concat in labels_concat]

        labels_concat = [
            label_concat / torch.maximum(num_positive, torch.tensor(1.0, device=num_positive.device))
            for label_concat, num_positive in zip(labels_concat, num_positives)
        ]

        obj_areas = [torch.sum(make_same_obj(pind, pind), dim=(2, 3)) for pind in [pind1, pind2]]

        weights = [
            torch.greater(num_positive[..., 0], 1e-3).to(torch.float32) / obj_area
            for num_positive, obj_area in zip(num_positives, obj_areas)
        ]

        logits_concat = [
            torch.cat([logits_dict["ab"], logits_dict["aa"]], dim=2).reshape((batch_size, num_samples, -1)),
            torch.cat([logits_dict["ba"], logits_dict["bb"]], dim=2).reshape((batch_size, num_samples, -1)),
        ]

        loss_a = manual_cross_entropy(logits_concat[0], labels_concat[0], weight=weights[0])
        loss_b = manual_cross_entropy(logits_concat[1], labels_concat[1], weight=weights[1])
        loss = loss_a + loss_b

        return loss
