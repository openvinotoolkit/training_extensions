# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``GPUAugmentationCallback``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback
from getitune.config.data import SubsetConfig
from getitune.data.augmentation.task_keys import DATA_KEYS_BY_TASK
from getitune.types.task import TaskType


def _subset_config() -> SubsetConfig:
    return SubsetConfig(
        batch_size=2,
        num_workers=0,
        augmentations_gpu=[{"class_path": "kornia.augmentation.RandomHorizontalFlip", "init_args": {"p": 1.0}}],
    )


@pytest.mark.parametrize("task", list(TaskType), ids=[t.value for t in TaskType])
def test_setup_builds_pipelines_with_shared_data_keys(task: TaskType) -> None:
    """Every pipeline should derive its data keys from the shared ``DATA_KEYS_BY_TASK``.

    This guards the migration away from the callback's private copy: if the
    callback ever reintroduces a hard-coded table, the keys fed to Kornia would
    diverge from ``DATA_KEYS_BY_TASK`` and this test would fail.
    """
    callback = GPUAugmentationCallback(train_config=_subset_config(), val_config=_subset_config())
    pl_module = MagicMock()
    pl_module.task = task

    captured: list[tuple[str, ...]] = []

    def fake_from_config(config: SubsetConfig, data_keys: list[str], sanitize_annotations: bool = False) -> MagicMock:
        captured.append(tuple(data_keys))
        return MagicMock()

    with patch(
        "getitune.backend.lightning.callbacks.gpu_augmentation.GPUAugmentationPipeline.from_config",
        side_effect=fake_from_config,
    ):
        callback.setup(MagicMock(), pl_module, "fit")

    expected = ("input", *DATA_KEYS_BY_TASK.get(task, ()))
    assert captured, "setup() did not build any augmentation pipeline"
    assert all(keys == expected for keys in captured)
