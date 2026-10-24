# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data augmentation scheduler for training.

Supports the CPU/GPU augmentation pipeline architecture:
- CPU augmentations (torchvision): run in Dataset workers before collate
- GPU augmentations (Kornia): run after batch transfer via GPUAugmentationCallback

Each policy defines ``augmentations_cpu`` and optionally ``augmentations_gpu``.
The scheduler swaps the CPU pipeline on the dataset at each ``__getitem__`` call
and swaps the GPU pipeline on the GPUAugmentationCallback at each epoch boundary.
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks.callback import Callback

from getitune.config.data import SubsetConfig
from getitune.data.augmentation import CPUAugmentationPipeline, GPUAugmentationPipeline

if TYPE_CHECKING:
    from multiprocessing import Value

    from lightning.pytorch import LightningModule, Trainer

    from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

logger = logging.getLogger(__name__)


class DataAugSwitch:
    """Dynamic augmentation policy switch for the CPU/GPU pipeline.

    Manages multiple augmentation policies and switches between them based on the
    current training epoch. Each policy contains separate CPU and GPU augmentation
    lists that are built into ``CPUAugmentationPipeline`` and ``GPUAugmentationPipeline``
    respectively.

    Args:
        policy_epochs (list[int]): List of 3 epoch boundaries ``[p0, p1]``:
            - ``epoch < p0``: ``no_aug``
            - ``p0 <= epoch < p1``: ``strong_aug_1`` or ``strong_aug_2`` (random)
            - ``epoch >= p1``: ``light_aug``
        policies (dict[str, dict[str, Any]]): Policy name → config mapping.
            Each config must have ``augmentations_cpu`` (list of transform dicts).
            Optionally ``augmentations_gpu`` (list of transform dicts).
        input_size (tuple[int,int] | list[int] | None): Model input size for
            ``$(input_size)`` placeholder resolution.

    Example::

        policies = {
            "no_aug": {
                "augmentations_cpu": [
                    {"class_path": "getitune.data.augmentation.transforms.Resize", ...},
                ],
                "augmentations_gpu": [
                    {"class_path": "kornia.augmentation.Normalize", ...},
                ],
            },
            "strong_aug_1": { ... },
            "strong_aug_2": { ... },
            "light_aug": { ... },
        }
        switch = DataAugSwitch([4, 29], policies, input_size=[640, 640])
    """

    def __init__(
        self,
        policy_epochs: list[int],
        policies: dict[str, dict[str, Any]],
        input_size: tuple[int, int] | list[int] | None = None,
    ) -> None:
        if len(policy_epochs) != 2:
            msg = "Expected 2 policy epochs for 3-stage scheduler (e.g., [4, 29])"
            raise ValueError(msg)

        self.policy_epochs = policy_epochs
        self._shared_epoch = None
        self.input_size = tuple(input_size) if input_size is not None else None
        self._gpu_pipeline_cache: dict[tuple[str, tuple[str, ...] | None], GPUAugmentationPipeline] = {}

        # Build pipelines for each policy
        self.policies: dict[str, dict[str, Any]] = {}
        for name, config in policies.items():
            cpu_aug_configs = config.get("augmentations_cpu", [])
            gpu_aug_configs = config.get("augmentations_gpu", [])

            # Build CPU pipeline via SubsetConfig → CPUAugmentationPipeline
            cpu_subset = SubsetConfig(
                augmentations_cpu=cpu_aug_configs,
                subset_name=name,
                input_size=self.input_size,  # type: ignore[arg-type]
            )
            cpu_pipeline = CPUAugmentationPipeline.from_config(cpu_subset)

            self.policies[name] = {
                "cpu_pipeline": cpu_pipeline,
                "gpu_aug_configs": gpu_aug_configs,
            }

    def set_shared_epoch(self, shared_epoch: Value) -> None:  # type: ignore[valid-type]
        """Set the shared multiprocessing epoch value."""
        self._shared_epoch = shared_epoch

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        if self._shared_epoch is None:
            msg = "Shared epoch not set. Call set_shared_epoch() first."
            raise ValueError(msg)
        return self._shared_epoch.value

    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set the current epoch."""
        if self._shared_epoch is None:
            msg = "Shared epoch not set. Call set_shared_epoch() first."
            raise ValueError(msg)
        self._shared_epoch.value = value

    @property
    def current_policy_name(self) -> str:
        """Get the current policy name based on epoch.

        During the strong augmentation phase (p0 <= epoch < p1), randomly
        selects between ``strong_aug_1`` and ``strong_aug_2`` so that each
        dataset worker can get a different variant per sample.
        """
        e = self.epoch
        p0, p1 = self.policy_epochs
        if e < p0:
            return "no_aug"
        if p0 <= e < p1:
            return secrets.choice(["strong_aug_1", "strong_aug_2"])
        return "light_aug"

    @property
    def current_gpu_policy_name(self) -> str:
        """Get deterministic policy name used for GPU pipeline selection."""
        e = self.epoch
        p0, p1 = self.policy_epochs
        if e < p0:
            return "no_aug"
        if p0 <= e < p1:
            if "strong_aug_1" in self.policies:
                return "strong_aug_1"
            if "strong_aug_2" in self.policies:
                return "strong_aug_2"
            return "no_aug"
        return "light_aug"

    def get_cpu_pipeline(self, policy_name: str) -> CPUAugmentationPipeline:
        """Get the CPU augmentation pipeline for a specific policy."""
        return self.policies[policy_name]["cpu_pipeline"]

    def get_gpu_aug_configs(self, policy_name: str) -> list[dict[str, Any]]:
        """Get raw GPU augmentation configs for a given policy.

        Returns empty list if no GPU augmentations are defined for that policy.
        """
        return self.policies[policy_name].get("gpu_aug_configs", [])

    def build_gpu_pipeline(self, policy_name: str, data_keys: list[str] | None = None) -> GPUAugmentationPipeline:
        """Build a GPUAugmentationPipeline for the given policy.

        Args:
            policy_name: Name of the policy.
            data_keys: Kornia data_keys for AugmentationSequential.

        Returns:
            GPUAugmentationPipeline instance.
        """
        cache_key = (policy_name, tuple(data_keys) if data_keys else None)
        if cache_key in self._gpu_pipeline_cache:
            return self._gpu_pipeline_cache[cache_key]

        gpu_configs = self.get_gpu_aug_configs(policy_name)
        if not gpu_configs:
            pipeline = GPUAugmentationPipeline([], data_keys=data_keys)
            self._gpu_pipeline_cache[cache_key] = pipeline
            return pipeline

        gpu_subset = SubsetConfig(
            augmentations_gpu=gpu_configs,
            batch_size=1,
            subset_name=policy_name,
            input_size=self.input_size,  # type: ignore[arg-type]
        )
        pipeline = GPUAugmentationPipeline.from_config(gpu_subset, data_keys=data_keys)
        self._gpu_pipeline_cache[cache_key] = pipeline
        return pipeline


class AugmentationSchedulerCallback(Callback):
    """Callback that drives augmentation policy switching at epoch boundaries.

    At each epoch start:
    1. Updates the shared epoch counter so CPU workers pick the correct policy.
    2. If GPU augmentations differ per policy, rebuilds the GPU pipeline on the
       ``GPUAugmentationCallback`` for the active policy.

    The CPU pipeline swap happens lazily in each dataset worker via
    ``DataAugSwitchMixin._apply_augmentation_switch()`` which calls
    ``DataAugSwitch.get_cpu_pipeline()``.

    Args:
        data_aug_switch: The DataAugSwitch that manages policies.
    """

    def __init__(self, data_aug_switch: DataAugSwitch | None = None):
        super().__init__()
        self.data_aug_switch = data_aug_switch
        self._gpu_aug_callback: GPUAugmentationCallback | None = None
        self._last_gpu_policy: str | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find and cache reference to GPUAugmentationCallback for GPU pipeline swaps."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        for callback in trainer.callbacks:  # type: ignore[union-attr]
            if isinstance(callback, GPUAugmentationCallback):
                self._gpu_aug_callback = callback
                break

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Update epoch and swap GPU pipeline if the phase changed."""
        if self.data_aug_switch is None:
            return

        self.data_aug_switch.epoch = trainer.current_epoch

        # Swap GPU pipeline if deterministic GPU phase changed.
        gpu_policy = self.data_aug_switch.current_gpu_policy_name
        if gpu_policy != self._last_gpu_policy:
            self._swap_gpu_pipeline(gpu_policy, pl_module)
            self._last_gpu_policy = gpu_policy

    def _swap_gpu_pipeline(self, policy_name: str, pl_module: LightningModule) -> None:
        """Rebuild and assign the GPU pipeline for the new policy."""
        if self._gpu_aug_callback is None or self.data_aug_switch is None:
            return

        # Get data_keys from the existing pipeline before overwriting
        data_keys = None
        if self._gpu_aug_callback._train_pipeline is not None:  # noqa: SLF001
            data_keys = self._gpu_aug_callback._train_pipeline.data_keys  # noqa: SLF001

        new_pipeline = self.data_aug_switch.build_gpu_pipeline(policy_name, data_keys=data_keys)
        # Move to same device as model
        new_pipeline = new_pipeline.to(pl_module.device)
        # Assign the new pipeline to the GPUAugmentationCallback
        self._gpu_aug_callback._train_pipeline = new_pipeline  # noqa: SLF001
        # Log the change. If the new pipeline has no augmentations, it will have aug_sequential = None.
        if new_pipeline.aug_sequential is not None:
            logger.info("Swapped GPU augmentation pipeline to policy '%s'", policy_name)
        else:
            logger.info("No GPU augmentations for policy '%s'; cleared GPU pipeline", policy_name)

    def set_data_aug_switch(self, data_aug_switch: DataAugSwitch) -> None:
        """Set or update the DataAugSwitch instance."""
        self.data_aug_switch = data_aug_switch
