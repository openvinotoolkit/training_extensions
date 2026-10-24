# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset mixins (CPU/GPU pipeline architecture)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from getitune.backend.lightning.callbacks.aug_scheduler import DataAugSwitch
from getitune.data.augmentation import CPUAugmentationPipeline
from getitune.data.dataset.mixins import DataAugSwitchMixin


class MockDataset(DataAugSwitchMixin):
    """Mock dataset class for testing the mixin."""

    def __init__(self):
        self.transforms: Any = None


class TestDataAugSwitchMixin:
    """Test cases for DataAugSwitchMixin with CPU/GPU pipeline."""

    @pytest.fixture
    def dataset(self):
        return MockDataset()

    @pytest.fixture
    def mock_switch(self):
        """A mock DataAugSwitch that returns predictable CPU pipeline."""
        s = MagicMock(spec=DataAugSwitch)
        s.current_policy_name = "no_aug"
        return s

    # -- lazy init -------------------------------------------------------

    def test_lazy_initialization(self, dataset):
        """Attribute should not exist until first access."""
        assert not hasattr(dataset, "data_aug_switch")
        assert not dataset.has_dynamic_augmentation
        assert hasattr(dataset, "data_aug_switch")
        assert dataset.data_aug_switch is None

    # -- set_data_aug_switch --------------------------------------------

    def test_set_data_aug_switch(self, dataset, mock_switch):
        dataset.set_data_aug_switch(mock_switch)
        assert dataset.data_aug_switch is mock_switch

    def test_set_data_aug_switch_replaces(self, dataset, mock_switch):
        dataset.set_data_aug_switch(mock_switch)
        new_switch = MagicMock(spec=DataAugSwitch)
        dataset.set_data_aug_switch(new_switch)
        assert dataset.data_aug_switch is new_switch

    # -- has_dynamic_augmentation ---------------------------------------

    def test_has_dynamic_false_when_none(self, dataset):
        assert not dataset.has_dynamic_augmentation

    def test_has_dynamic_true_when_set(self, dataset, mock_switch):
        dataset.set_data_aug_switch(mock_switch)
        assert dataset.has_dynamic_augmentation

    # -- _apply_augmentation_switch -------------------------------------

    def test_apply_returns_none_when_no_switch(self, dataset):
        result = dataset._apply_augmentation_switch()
        assert result is None
        assert dataset.transforms is None

    def test_apply_sets_transforms_to_cpu_pipeline(self, dataset, mock_switch):
        expected_pipeline = MagicMock(spec=CPUAugmentationPipeline)
        mock_switch.get_cpu_pipeline.return_value = expected_pipeline
        dataset.set_data_aug_switch(mock_switch)
        policy = dataset._apply_augmentation_switch()
        assert policy == "no_aug"
        assert dataset.transforms is expected_pipeline
        mock_switch.get_cpu_pipeline.assert_called_once_with("no_aug")

    def test_apply_follows_policy_changes(self, dataset, mock_switch):
        dataset.set_data_aug_switch(mock_switch)

        # First call → no_aug
        pipeline_no_aug = MagicMock(spec=CPUAugmentationPipeline)
        mock_switch.current_policy_name = "no_aug"
        mock_switch.get_cpu_pipeline.return_value = pipeline_no_aug
        dataset._apply_augmentation_switch()
        assert dataset.transforms is pipeline_no_aug

        # Policy changes → strong_aug_1
        pipeline_strong = MagicMock(spec=CPUAugmentationPipeline)
        mock_switch.current_policy_name = "strong_aug_1"
        mock_switch.get_cpu_pipeline.return_value = pipeline_strong
        policy = dataset._apply_augmentation_switch()
        assert policy == "strong_aug_1"
        assert dataset.transforms is pipeline_strong

    def test_apply_does_not_touch_to_tv_image(self, dataset, mock_switch):
        """to_tv_image should NOT be mutated — GPU pipeline handles normalization."""
        dataset.to_tv_image = True
        dataset.set_data_aug_switch(mock_switch)
        dataset._apply_augmentation_switch()
        assert dataset.to_tv_image is True

    # -- edge cases -----------------------------------------------------

    def test_mixin_on_plain_class(self):
        """Mixin works even on a plain class that doesn't inherit VisionDataset."""

        class PlainDataset(DataAugSwitchMixin):
            def __init__(self):
                self.transforms: Any = None

        ds = PlainDataset()
        assert not ds.has_dynamic_augmentation

        mock = MagicMock(spec=DataAugSwitch)
        mock.current_policy_name = "light_aug"
        expected_pipeline = MagicMock(spec=CPUAugmentationPipeline)
        mock.get_cpu_pipeline.return_value = expected_pipeline
        ds.set_data_aug_switch(mock)

        assert ds.has_dynamic_augmentation
        policy = ds._apply_augmentation_switch()
        assert policy == "light_aug"
        assert ds.transforms is expected_pipeline
