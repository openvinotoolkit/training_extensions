# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.task_modules.samplers.base_sampler.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/samplers/base_sampler.py
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
from otx.algo.common.utils.structures import AssignResult, SamplingResult

if TYPE_CHECKING:
    from otx.algo.utils.mmengine_utils import InstanceData


def ensure_rng(rng: int | np.random.RandomState | None = None) -> np.random.RandomState:
    """Coerces input into a random number generator.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/utils/util_random.py#L6-L34

    If the input is None, then a global random state is returned.

    If the input is a numeric value, then that is used as a seed to construct a
    random state. Otherwise the input is returned as-is.

    Adapted from [1]_.

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class

    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        .. [1] https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270  # noqa: E501
    """
    if rng is None:
        return np.random.mtrand._rand  # noqa: SLF001
    if isinstance(rng, int):
        return np.random.RandomState(rng)
    return rng


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_ub (int): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(
        self,
        num: int,
        pos_fraction: float,
        neg_pos_ub: int = -1,
        add_gt_as_proposals: bool = True,
        **kwargs,
    ) -> None:
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result: AssignResult, num_expected: int, **kwargs) -> torch.Tensor:
        """Sample positive samples."""

    @abstractmethod
    def _sample_neg(self, assign_result: AssignResult, num_expected: int, **kwargs) -> torch.Tensor:
        """Sample negative samples."""

    @abstractmethod
    def sample(
        self,
        assign_result: AssignResult,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        **kwargs,
    ) -> SamplingResult:
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (AssignResult): Assigning results.
            pred_instances (InstanceData): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the `InstanceData`
                in other places.
            gt_instances (InstanceData): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).

        Returns:
            SamplingResult: Sampling result.
        """


class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs) -> None:
        pass

    def _sample_pos(self, assign_result: AssignResult, num_expected: int, **kwargs) -> torch.Tensor:
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, assign_result: AssignResult, num_expected: int, **kwargs) -> torch.Tensor:
        """Sample negative samples."""
        raise NotImplementedError

    def sample(
        self,
        assign_result: AssignResult,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        *args,
        **kwargs,
    ) -> SamplingResult:
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (AssignResult): Bbox assigning results.
            pred_instances (InstanceData): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (InstanceData): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            SamplingResult: sampler results
        """
        gt_bboxes = gt_instances.bboxes  # type: ignore[attr-defined]
        priors = pred_instances.priors  # type: ignore[attr-defined]

        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)
        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False,
        )


class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_ub (int): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(
        self,
        num: int,
        pos_fraction: float,
        neg_pos_ub: int = -1,
        add_gt_as_proposals: bool = True,
        **kwargs,
    ):
        super().__init__(
            num=num,
            pos_fraction=pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals,
        )
        self.rng = ensure_rng(kwargs.get("rng", None))

    def random_choice(self, gallery: torch.Tensor | np.ndarray | list, num: int) -> torch.Tensor | np.ndarray:
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
        _gallery: torch.Tensor = torch.tensor(gallery, dtype=torch.long, device=device) if not is_tensor else gallery
        perm = torch.randperm(_gallery.numel())[:num].to(device=_gallery.device)
        rand_inds = _gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result: AssignResult, num_expected: int, **kwargs: dict) -> torch.Tensor | np.ndarray:
        """Randomly sample some positive samples.

        Args:
            assign_result (AssignResult): Bbox assigning results.
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

    def _sample_neg(self, assign_result: AssignResult, num_expected: int, **kwargs: dict) -> torch.Tensor | np.ndarray:
        """Randomly sample some negative samples.

        Args:
            assign_result (AssignResult): Bbox assigning results.
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

    def sample(
        self,
        assign_result: AssignResult,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        **kwargs,
    ) -> SamplingResult:
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (AssignResult): Assigning results.
            pred_instances (InstanceData): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the `InstanceData`
                in other places.
            gt_instances (InstanceData): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).

        Returns:
            SamplingResult: Sampling result.
        """
        gt_bboxes = gt_instances.bboxes  # type: ignore[attr-defined]
        priors = pred_instances.priors  # type: ignore[attr-defined]
        gt_labels = gt_instances.labels  # type: ignore[attr-defined]
        if len(priors.shape) < 2:
            priors = priors[None, :]

        gt_flags = priors.new_zeros((priors.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            priors = torch.cat([gt_bboxes, priors], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = priors.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result, num_expected_pos, bboxes=priors, **kwargs)  # noqa: SLF001
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(assign_result, num_expected_neg, bboxes=priors, **kwargs)  # noqa: SLF001
        neg_inds = neg_inds.unique()

        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
        )
