"""Unit tests for segmentation dataset."""

from __future__ import annotations

from unittest.mock import Mock

from datumaro.experimental import Dataset

from getitune.data.dataset.segmentation import SegmentationDataset
from getitune.data.entity.sample import SegmentationSample


class TestSegmentationDataset:
    def setup_method(self):
        # Mock Datumaro experimental Dataset subset
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=3)
        self.mock_dm_subset.convert_to_schema = Mock(return_value=self.mock_dm_subset)

        # Mock schema attributes for masks label_info
        mock_schema = Mock()
        mock_attributes = {"masks": Mock()}
        mock_attributes["masks"].categories = Mock()
        mock_attributes["masks"].categories.labels = ["bg", "person", "car"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        self.mock_transforms = Mock()

    def test_init_converts_schema_and_sets_label_info(self):
        dataset = SegmentationDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
            ignore_index=255,
        )

        # Ensure schema conversion to SegmentationSample
        self.mock_dm_subset.convert_to_schema.assert_called_once_with(SegmentationSample)
        # Ensure label_info populated from masks labels
        assert dataset.label_info.label_names == ["bg", "person", "car"]
        # Task type
        assert dataset.task_type.name == "SEMANTIC_SEGMENTATION"
