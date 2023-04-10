"""Hook module to track loss dynamics during training and export these statistics to Datumaro format."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from typing import Any, Dict, List

from otx.api.entities.datasets import DatasetEntity

__all__ = ["LossDynamicsTracker", "LossDynamicsTrackingMixin"]


class LossDynamicsTracker:
    """Class to track loss dynamics and export it to Datumaro format."""

    def __init__(self) -> None:
        self.initialized = False

    def init_with_otx_dataset(self, otx_dataset: DatasetEntity) -> None:
        """DatasetEntity should be injected to the tracker for the initialization."""
        self._loss_dynamics: Dict[Any, List] = defaultdict(list)
        self.initialized = True

    def accumulate(self, outputs, iter) -> None:
        """Accumulate training loss dynamics for each training step."""
        raise NotImplementedError()

    def export(self, output_path: str) -> None:
        """Export loss dynamics statistics to Datumaro format."""
        raise NotImplementedError()


class LossDynamicsTrackingMixin:
    """Mix-in to track loss dynamics during training."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loss_dyns_tracker = LossDynamicsTracker()

    @property
    def loss_dyns_tracker(self) -> LossDynamicsTracker:
        """Get tracker."""
        return self._loss_dyns_tracker
