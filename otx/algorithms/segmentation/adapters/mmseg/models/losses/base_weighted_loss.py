"""Base weighted loss function for semantic segmentation."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABCMeta, abstractmethod

import torch
from mmseg.core import build_pixel_sampler
from scipy.special import erfinv  # pylint: disable=no-name-in-module
from torch import nn

from otx.algorithms.segmentation.adapters.mmseg.utils.builder import (
    build_scalar_scheduler,
)


# pylint: disable=too-many-instance-attributes, unused-argument
class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.

    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.

    Args:
        loss_weight (float or dict): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        ignore_index=255,
        sampler=None,
        loss_jitter_prob=None,
        loss_jitter_momentum=0.1,
        **kwargs
    ):
        super().__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

        self.sampler = None
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, ignore_index=ignore_index)

        self._smooth_loss = None
        self._jitter_sigma_factor = None
        self._loss_jitter_momentum = loss_jitter_momentum
        assert 0.0 < self._loss_jitter_momentum < 1.0
        if loss_jitter_prob is not None:
            assert 0.0 < loss_jitter_prob < 1.0
            self._jitter_sigma_factor = 1.0 / ((2.0**0.5) * erfinv(1.0 - 2.0 * loss_jitter_prob))

        self._loss_weight_scheduler = build_scalar_scheduler(loss_weight, default_value=1.0)

        self._iter = 0
        self._last_loss_weight = 0
        self._epoch_size = 1

    def set_step_params(self, init_iter, epoch_size):
        """Set step parameters."""
        assert init_iter >= 0
        assert epoch_size > 0

        self._iter = init_iter
        self._epoch_size = epoch_size

    @property
    def with_loss_jitter(self):
        """Check loss jitter."""
        return self._jitter_sigma_factor is not None

    @property
    def iter(self):
        """Return iteration."""
        return self._iter

    @property
    def epoch_size(self):
        """Return epoch size."""
        return self._epoch_size

    @property
    def last_loss_weight(self):
        """Return last loss weight."""
        return self._last_loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.

        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.

        Returns:
            torch.Tensor: The calculated loss.
        """

        loss, meta = self._forward(*args, **kwargs)
        # make sure meta data are tensor as well for aggregation
        # when parsing loss in sgementator
        for key, val in meta.items():
            meta[key] = torch.tensor(val, dtype=loss.dtype, device=loss.device)

        if self.with_loss_jitter and loss.numel() == 1:
            if self._smooth_loss is None:
                self._smooth_loss = loss.item()
            else:
                self._smooth_loss = (
                    1.0 - self._loss_jitter_momentum
                ) * self._smooth_loss + self._loss_jitter_momentum * loss.item()

            jitter_sigma = self._jitter_sigma_factor * abs(self._smooth_loss)
            jitter_point = torch.normal(0.0, jitter_sigma, [], device=loss.device, dtype=loss.dtype)
            loss = (loss - jitter_point).abs() + jitter_point

        self._last_loss_weight = self._loss_weight_scheduler(self.iter, self.epoch_size)
        out_loss = self._last_loss_weight * loss

        self._iter += 1

        return out_loss, meta
