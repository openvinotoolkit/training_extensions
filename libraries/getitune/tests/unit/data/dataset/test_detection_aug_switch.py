# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for DetectionDataset with DataAugSwitchMixin."""

from multiprocessing import Value
from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.callbacks.aug_scheduler import DataAugSwitch
from getitune.data.dataset.detection import DetectionDataset
from getitune.data.dataset.mixins import DataAugSwitchMixin


class TestDetectionDatasetWithAugSwitch:
    """Integration tests for DetectionDataset with DataAugSwitchMixin."""

    @pytest.fixture
    def sample_policies(self):
        """Create sample augmentation policies."""
        return {
            "no_aug": {
                "augmentations_cpu": [
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
                    },
                ],
            },
            "strong_aug_1": {
                "augmentations_cpu": [
                    {"class_path": "torchvision.transforms.v2.RandomHorizontalFlip", "init_args": {"p": 0.5}},
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
                    },
                ],
            },
            "strong_aug_2": {
                "augmentations_cpu": [
                    {"class_path": "torchvision.transforms.v2.RandomVerticalFlip", "init_args": {"p": 0.5}},
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
                    },
                ],
            },
            "light_aug": {
                "augmentations_cpu": [
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": [640, 640], "keep_aspect_ratio": False},
                    },
                ],
            },
        }

    @pytest.fixture
    def data_aug_switch(self, sample_policies):
        """Create a DataAugSwitch instance."""
        switch = DataAugSwitch([4, 29], sample_policies)
        shared_epoch = Value("i", 0)
        switch.set_shared_epoch(shared_epoch)
        return switch

    @pytest.fixture
    def mock_dm_subset(self):
        """Create a mock datumaro subset."""
        mock_subset = MagicMock()
        mock_subset.categories = MagicMock()
        mock_subset.categories.return_value = []
        mock_subset.__len__ = MagicMock(return_value=10)

        # Mock items for iteration
        mock_items = []
        for i in range(10):
            mock_item = MagicMock()
            mock_item.id = f"item_{i}"
            mock_item.media = MagicMock()
            mock_item.media.data = MagicMock()
            mock_item.annotations = []
            mock_items.append(mock_item)

        mock_subset.__iter__ = MagicMock(return_value=iter(mock_items))
        return mock_subset

    @pytest.fixture
    def detection_dataset(self, mock_dm_subset):
        """Create an DetectionDataset instance."""
        return DetectionDataset(
            dm_subset=mock_dm_subset,
            transforms=None,
        )

    def test_detection_dataset_inherits_mixin(self, detection_dataset):
        """Test that DetectionDataset inherits from DataAugSwitchMixin."""
        assert isinstance(detection_dataset, DataAugSwitchMixin)

    def test_detection_dataset_mixin_initialization(self, detection_dataset):
        """Test that mixin is properly initialized."""
        # Initially, the attribute shouldn't exist due to lazy initialization
        assert not hasattr(detection_dataset, "data_aug_switch")

        # After calling has_dynamic_augmentation, it should be initialized
        assert not detection_dataset.has_dynamic_augmentation
        assert hasattr(detection_dataset, "data_aug_switch")
        assert detection_dataset.data_aug_switch is None

    def test_set_data_aug_switch_on_detection_dataset(self, detection_dataset, data_aug_switch):
        """Test setting DataAugSwitch on detection dataset."""
        detection_dataset.set_data_aug_switch(data_aug_switch)

        assert detection_dataset.data_aug_switch is data_aug_switch
        assert detection_dataset.has_dynamic_augmentation

    def test_augmentation_switch_integration_no_aug_stage(self, detection_dataset, data_aug_switch):
        """Test augmentation switch integration in no_aug stage."""
        data_aug_switch.epoch = 2  # no_aug stage
        detection_dataset.set_data_aug_switch(data_aug_switch)

        # Mock the _get_item_impl method to test the integration
        with patch.object(detection_dataset, "_get_item_impl") as mock_get_item:
            mock_entity = MagicMock()
            mock_get_item.return_value = mock_entity

            with patch.object(detection_dataset, "_apply_transforms") as mock_apply_transforms:
                mock_apply_transforms.return_value = mock_entity

                # This should trigger the augmentation switch
                detection_dataset._get_item_impl(0)

                # Check that the policy was applied
                assert data_aug_switch.current_policy_name == "no_aug"

    def test_augmentation_switch_integration_strong_aug_stage(self, detection_dataset, data_aug_switch):
        """Test augmentation switch integration in strong_aug stage."""
        data_aug_switch.epoch = 15  # strong_aug stage
        detection_dataset.set_data_aug_switch(data_aug_switch)

        with patch.object(detection_dataset, "_get_item_impl") as mock_get_item:
            mock_entity = MagicMock()
            mock_get_item.return_value = mock_entity

            with patch.object(detection_dataset, "_apply_transforms") as mock_apply_transforms:
                mock_apply_transforms.return_value = mock_entity

                # This should trigger the augmentation switch
                detection_dataset._get_item_impl(0)

                # Check that the policy was applied
                policy_name = data_aug_switch.current_policy_name
                assert policy_name in ["strong_aug_1", "strong_aug_2"]

    def test_augmentation_switch_integration_light_aug_stage(self, detection_dataset, data_aug_switch):
        """Test augmentation switch integration in light_aug stage."""
        data_aug_switch.epoch = 35  # light_aug stage
        detection_dataset.set_data_aug_switch(data_aug_switch)

        with patch.object(detection_dataset, "_get_item_impl") as mock_get_item:
            mock_entity = MagicMock()
            mock_get_item.return_value = mock_entity

            with patch.object(detection_dataset, "_apply_transforms") as mock_apply_transforms:
                mock_apply_transforms.return_value = mock_entity

                # This should trigger the augmentation switch
                detection_dataset._get_item_impl(0)

                # Check that the policy was applied
                assert data_aug_switch.current_policy_name == "light_aug"

    def test_transforms_updated_correctly(self, detection_dataset, data_aug_switch):
        """Test that transforms are updated correctly when epoch changes."""
        from getitune.data.augmentation import CPUAugmentationPipeline

        detection_dataset.set_data_aug_switch(data_aug_switch)

        # Test different epochs and verify transforms update
        test_epochs = [2, 15, 35]
        expected_policies = ["no_aug", "strong_aug", "light_aug"]

        for epoch, expected_policy_type in zip(test_epochs, expected_policies):
            data_aug_switch.epoch = epoch

            # Apply augmentation switch
            policy_name = detection_dataset._apply_augmentation_switch()

            # Check that transforms were updated
            if expected_policy_type == "strong_aug":
                assert policy_name in ["strong_aug_1", "strong_aug_2"]
            else:
                assert policy_name == expected_policy_type

            # Verify that the transforms is now the CPU pipeline for this policy
            assert isinstance(detection_dataset.transforms, CPUAugmentationPipeline)

    def test_detection_dataset_without_aug_switch(self, detection_dataset):
        """Test that detection dataset works normally without augmentation switch."""

        # Store original transforms
        original_transforms = detection_dataset.transforms

        # Apply augmentation switch (should do nothing)
        detection_dataset._apply_augmentation_switch()

        # Verify nothing changed
        assert detection_dataset.transforms == original_transforms

    def test_epoch_boundary_conditions(self, detection_dataset, data_aug_switch):
        """Test epoch boundary conditions."""
        detection_dataset.set_data_aug_switch(data_aug_switch)

        # Test boundary epochs
        boundary_tests = [
            (3, "no_aug"),
            (4, "strong_aug"),
            (28, "strong_aug"),
            (29, "light_aug"),
            (50, "light_aug"),
        ]

        for epoch, expected_stage in boundary_tests:
            data_aug_switch.epoch = epoch

            detection_dataset._apply_augmentation_switch()

            current_policy = data_aug_switch.current_policy_name
            if expected_stage == "strong_aug":
                assert current_policy in ["strong_aug_1", "strong_aug_2"]
            else:
                assert current_policy == expected_stage

    def test_multiple_datasets_same_switch(self, mock_dm_subset, data_aug_switch):
        """Test multiple datasets sharing the same augmentation switch."""
        # Create multiple datasets
        dataset1 = DetectionDataset(dm_subset=mock_dm_subset, transforms=None)
        dataset2 = DetectionDataset(dm_subset=mock_dm_subset, transforms=None)

        # Set the same switch on both
        dataset1.set_data_aug_switch(data_aug_switch)
        dataset2.set_data_aug_switch(data_aug_switch)

        # Change epoch
        data_aug_switch.epoch = 50

        # Both datasets should see the same policy
        policy1 = dataset1._apply_augmentation_switch()
        transforms1 = dataset1.transforms

        policy2 = dataset2._apply_augmentation_switch()
        transforms2 = dataset2.transforms

        # Both should have the same policy and transforms
        assert policy1 == policy2
        assert transforms1 == transforms2

    def test_error_handling_without_shared_epoch(self, detection_dataset, sample_policies):
        """Test error handling when DataAugSwitch doesn't have shared epoch set."""
        # Create switch without shared epoch
        switch = DataAugSwitch([4, 29], sample_policies)
        detection_dataset.set_data_aug_switch(switch)

        # This should raise an error when trying to access current_policy_name
        with pytest.raises(ValueError, match="Shared epoch not set"):
            detection_dataset._apply_augmentation_switch()

    def test_type_annotations_compatibility(self, detection_dataset):
        """Test that type annotations work correctly with mixin."""
        # This test ensures the type: ignore[misc] comment is working
        assert isinstance(detection_dataset, DetectionDataset)
        assert isinstance(detection_dataset, DataAugSwitchMixin)

        # Test that all mixin methods are available
        assert callable(detection_dataset.set_data_aug_switch)
        assert callable(detection_dataset._apply_augmentation_switch)
        assert isinstance(detection_dataset.has_dynamic_augmentation, bool)

    def test_transforms_pipeline_switch(self, detection_dataset, data_aug_switch, mocker):
        """Test that augmentation switch is properly set up on detection dataset."""
        detection_dataset.set_data_aug_switch(data_aug_switch)

        # Verify that the augmentation switch is set
        assert detection_dataset.data_aug_switch is data_aug_switch
        assert detection_dataset.has_dynamic_augmentation is True
