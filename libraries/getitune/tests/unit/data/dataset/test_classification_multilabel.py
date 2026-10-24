"""Unit tests for multi-label classification dataset."""

from __future__ import annotations

from unittest.mock import Mock

import polars as pl
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors

from getitune.data.dataset.classification import MultilabelClsDataset
from getitune.data.entity.sample import ClassificationMultiLabelSample


class TestMultilabelClsDataset:
    def setup_method(self):
        # Mock Datumaro experimental Dataset subset
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=10)
        self.mock_dm_subset.convert_to_schema = Mock(return_value=self.mock_dm_subset)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        mock_attributes["label"].categories.labels = ["class_0", "class_1", "class_2", "class_3"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        # Mock label_categories used by the dataset implementation
        mock_label_categories = Mock()
        mock_label_categories.labels = ["class_0", "class_1", "class_2", "class_3"]
        self.mock_dm_subset.label_categories = mock_label_categories

        self.mock_transforms = Mock()

    def test_init_converts_schema_and_sets_label_info(self):
        dataset = MultilabelClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        # Ensure we convert to the expected schema
        self.mock_dm_subset.convert_to_schema.assert_called_once_with(ClassificationMultiLabelSample)
        assert dataset.num_classes == 4
        assert dataset.label_info.label_names == ["class_0", "class_1", "class_2", "class_3"]

    def test_convert_to_onehot_and_ignored_labels(self):
        dataset = MultilabelClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        # Empty labels (float) must produce all-zero vector
        labels = torch.as_tensor([], dtype=torch.float32)
        onehot = dataset._convert_to_onehot(labels, ignored_labels=[])
        assert torch.equal(onehot, torch.zeros(dataset.num_classes, dtype=onehot.dtype))

        # Non-empty with ignored label should set that index to -1
        labels2 = torch.as_tensor([0, 2])
        onehot2 = dataset._convert_to_onehot(labels2, ignored_labels=[1])
        assert onehot2[0] == 1
        assert onehot2[2] == 1
        assert onehot2[1] == -1

    def test_get_idx_list_per_classes_int_and_string(self):
        # Prepare three items with multilabel indices
        mock_items = []
        for lbls in ([0, 1], [2], [1, 3], [0, 2]):
            mock_item = Mock()
            mock_item.label.tolist.return_value = list(lbls)
            mock_items.append(mock_item)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = MultilabelClsDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        dataset.dm_subset.__len__ = Mock(return_value=4)

        # Integer mapping
        result_int = dataset.get_idx_list_per_classes()
        assert result_int == {0: [0, 3], 1: [0, 2], 2: [1, 3], 3: [2]}

        # String mapping
        # Reset side_effect for second pass
        self.mock_dm_subset.__getitem__.side_effect = mock_items
        result_str = dataset.get_idx_list_per_classes(use_string_label=True)
        assert result_str == {
            "class_0": [0, 3],
            "class_1": [0, 2],
            "class_2": [1, 3],
            "class_3": [2],
        }


class TestMultilabelClsDatasetLabelRange:
    """Test that multi-label class indices are not limited to the UInt8 range."""

    def test_label_field_is_uint16(self) -> None:
        assert ClassificationMultiLabelSample.label.dtype == pl.UInt16()  # type: ignore[attr-defined]

    def test_labels_above_255(self) -> None:
        num_classes = 300
        labels = [[0, 255], [256, num_classes - 1]]
        categories = {"label": LabelCategories(labels=tuple(f"class_{i}" for i in range(num_classes)))}
        dm_dataset = Dataset(ClassificationMultiLabelSample, categories=categories)  # type: ignore[arg-type]
        for sample_labels in labels:
            dm_dataset.append(
                ClassificationMultiLabelSample(
                    image=tv_tensors.Image(torch.zeros(3, 8, 8, dtype=torch.uint8)),
                    label=torch.tensor(sample_labels, dtype=torch.uint16),
                    dm_image_info=DmImageInfo(width=8, height=8),
                    subset=Subset.TRAINING,
                ),
            )
        dataset = MultilabelClsDataset(dm_subset=dm_dataset, transforms=[])
        assert dataset.num_classes == num_classes
        for idx, sample_labels in enumerate(labels):
            onehot = dataset[idx].label  # type: ignore[attr-defined]
            assert onehot.shape == (num_classes,)
            assert onehot.nonzero().flatten().tolist() == sample_labels
        assert dataset.get_idx_list_per_classes() == {0: [0], 255: [0], 256: [1], num_classes - 1: [1]}
