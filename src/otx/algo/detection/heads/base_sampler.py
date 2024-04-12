# Copyright (c) OpenMMLab. All rights reserved.
"""Base Sampler implementation from mmdet."""

from abc import ABCMeta, abstractmethod

import torch
from mmengine.structures import InstanceData

from otx.algo.detection.utils.structures import AssignResult, SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int): Upper bound number of negative and
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
            assign_result (:obj:`AssignResult`): Assigning results.
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if len(priors.shape) < 2:
            priors = priors[None, :]

        gt_flags = priors.new_zeros((priors.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            gt_bboxes_ = gt_bboxes
            priors = torch.cat([gt_bboxes_, priors], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = priors.new_ones(gt_bboxes_.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(  # noqa: SLF001
            assign_result,
            num_expected_pos,
            bboxes=priors,
            **kwargs,
        )
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
        neg_inds = self.neg_sampler._sample_neg(  # noqa: SLF001
            assign_result,
            num_expected_neg,
            bboxes=priors,
            **kwargs,
        )
        neg_inds = neg_inds.unique()

        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
        )


class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
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
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors

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
