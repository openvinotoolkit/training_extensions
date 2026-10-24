# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base prior generator."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from torch import Tensor


class BasePriorGenerator(metaclass=ABCMeta):
    """Base class for prior generator."""

    strides: list[tuple[int, int]]
    grid_anchors: Callable[..., list[Tensor]]

    @property
    @abstractmethod
    def num_base_priors(self) -> list[int]:
        """Return the number of priors (anchors/points) at a point on the feature grid."""

    @property
    @abstractmethod
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied."""

    @abstractmethod
    def grid_priors(self, *args, **kwargs) -> list[Tensor]:
        """Generate grid anchors/points of multiple feature levels."""

    @abstractmethod
    def valid_flags(self, *args, **kwargs) -> list[Tensor]:
        """Generate valid flags of anchors/points of multiple feature levels."""
