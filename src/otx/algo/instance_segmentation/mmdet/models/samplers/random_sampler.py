"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch
from mmengine.registry import TASK_UTILS
from numpy import ndarray
from torch import Tensor

from otx.algo.instance_segmentation.mmdet.models.assigners import AssignResult
from otx.algo.instance_segmentation.mmdet.models.samplers.base_sampler import BaseSampler


@TASK_UTILS.register_module()
class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self, num: int, pos_fraction: float, neg_pos_ub: int = -1, add_gt_as_proposals: bool = True, **kwargs):
        from .sampling_result import ensure_rng

        super().__init__(
            num=num,
            pos_fraction=pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals,
        )
        self.rng = ensure_rng(kwargs.get("rng", None))

    def random_choice(self, gallery: Tensor | ndarray | list, num: int) -> Tensor | ndarray:
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        if len(gallery) < num:
            msg = f"Cannot sample {num} elements from a set of size {len(gallery)}"
            raise ValueError(msg)

        is_tensor = isinstance(gallery, torch.Tensor)
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        _gallery: Tensor = torch.tensor(gallery, dtype=torch.long, device=device) if not is_tensor else gallery
        perm = torch.randperm(_gallery.numel())[:num].to(device=_gallery.device)
        rand_inds = _gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result: AssignResult, num_expected: int, **kwargs: dict) -> Tensor | ndarray:
        """Randomly sample some positive samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result: AssignResult, num_expected: int, **kwargs: dict) -> Tensor | ndarray:
        """Randomly sample some negative samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        return self.random_choice(neg_inds, num_expected)
