# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixins for getitune datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from getitune.backend.lightning.callbacks.aug_scheduler import DataAugSwitch


class DataAugSwitchMixin:
    """Mixin that provides dynamic augmentation switching for the CPU/GPU pipeline.

    At each ``__getitem__`` call the dataset checks the current policy from
    ``DataAugSwitch`` and swaps ``self.transforms`` to the corresponding
    ``CPUAugmentationPipeline``. GPU augmentations are swapped separately by
    ``AugmentationSchedulerCallback`` at epoch boundaries.

    Usage::

        class MyDataset(VisionDataset, DataAugSwitchMixin):
            def _apply_transforms(self, entity):
                if self.has_dynamic_augmentation:
                    self._apply_augmentation_switch()
                return super()._apply_transforms(entity)
    """

    def _ensure_data_aug_switch_initialized(self) -> None:
        """Ensure data_aug_switch attribute is initialized."""
        if not hasattr(self, "data_aug_switch"):
            self.data_aug_switch: DataAugSwitch | None = None

    def set_data_aug_switch(self, data_aug_switch: DataAugSwitch) -> None:
        """Set data augmentation switch.

        Args:
            data_aug_switch: DataAugSwitch instance that manages dynamic augmentation policies.
        """
        self._ensure_data_aug_switch_initialized()
        self.data_aug_switch = data_aug_switch

    def _apply_augmentation_switch(self) -> str | None:
        """Swap ``self.transforms`` to the active policy's CPU pipeline.

        Called before ``_apply_transforms`` in each ``__getitem__``.
        Only swaps the CPU pipeline; GPU pipeline is handled by the callback.

        Returns:
            Policy name, or None if no switch is configured.
        """
        self._ensure_data_aug_switch_initialized()
        if self.data_aug_switch is None:
            return None
        policy_name = self.data_aug_switch.current_policy_name
        self.transforms = self.data_aug_switch.get_cpu_pipeline(policy_name)
        return policy_name

    @property
    def has_dynamic_augmentation(self) -> bool:
        """Check if dynamic augmentation is available and configured."""
        self._ensure_data_aug_switch_initialized()
        return self.data_aug_switch is not None
