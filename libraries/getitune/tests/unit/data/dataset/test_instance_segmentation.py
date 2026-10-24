"""Unit tests for instance segmentation dataset."""

from __future__ import annotations

from unittest.mock import Mock

import polars as pl
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import Colormap, LabelCategories, MaskCategories, RgbColor
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors

from getitune.data.dataset.base import _default_collate_fn
from getitune.data.dataset.instance_segmentation import InstanceSegDataset
from getitune.data.entity.sample import InstanceSegmentationSample


class TestInstanceSegDataset:
    def setup_method(self):
        # Mock Datumaro experimental Dataset subset
        self.mock_dm_subset = Mock(spec=Dataset)
        self.mock_dm_subset.__len__ = Mock(return_value=5)
        self.mock_dm_subset.convert_to_schema = Mock(return_value=self.mock_dm_subset)

        # Mock schema attributes for label_info
        mock_schema = Mock()
        mock_attributes = {"label": Mock()}
        mock_attributes["label"].categories = Mock()
        mock_attributes["label"].categories.labels = ["person", "car", "dog"]
        mock_schema.attributes = mock_attributes
        self.mock_dm_subset.schema = mock_schema

        # Mock label_categories used by the dataset implementation
        mock_label_categories = Mock()
        mock_label_categories.labels = ["person", "car", "dog"]
        self.mock_dm_subset.label_categories = mock_label_categories

        self.mock_transforms = Mock()

    def test_get_idx_list_per_classes_int_and_string(self):
        # Prepare items with multi-instance labels per sample
        mock_items = []
        for lbls in ([0, 1], [2], [1, 2], [0]):
            mock_item = Mock()
            mock_item.label.tolist.return_value = list(lbls)
            mock_items.append(mock_item)

        self.mock_dm_subset.__getitem__ = Mock(side_effect=mock_items)

        dataset = InstanceSegDataset(
            dm_subset=self.mock_dm_subset,
            transforms=self.mock_transforms,
        )

        dataset.dm_subset.__len__ = Mock(return_value=4)

        # Integer mapping
        result_int = dataset.get_idx_list_per_classes()
        assert result_int == {0: [0, 3], 1: [0, 2], 2: [1, 2]}

        # String mapping
        # Reset side_effect for second pass
        self.mock_dm_subset.__getitem__.side_effect = mock_items
        result_str = dataset.get_idx_list_per_classes(use_string_label=True)
        assert result_str == {
            "person": [0, 3],
            "car": [0, 2],
            "dog": [1, 2],
        }


class TestInstanceSegDatasetLabelRange:
    """Test that instance segmentation labels are not limited to the UInt8 range."""

    def test_label_field_is_uint16(self) -> None:
        assert InstanceSegmentationSample.label.dtype == pl.UInt16()  # type: ignore[attr-defined]

    def test_labels_above_255(self) -> None:
        num_classes = 300
        labels = [[0, 255], [256, num_classes - 1], []]
        categories = {
            "label": LabelCategories(labels=tuple(f"class_{i}" for i in range(num_classes))),
            "masks": MaskCategories(
                labels=["background", "instance"],
                colormap=Colormap({0: RgbColor(0, 0, 0), 1: RgbColor(255, 255, 255)}),
            ),
        }
        dm_dataset = Dataset(InstanceSegmentationSample, categories=categories)  # type: ignore[arg-type]
        for sample_labels in labels:
            n = len(sample_labels)
            masks = torch.zeros(n, 8, 8, dtype=torch.uint8)
            masks[:, :4, :4] = 1
            dm_dataset.append(
                InstanceSegmentationSample(
                    image=tv_tensors.Image(torch.zeros(3, 8, 8, dtype=torch.uint8)),
                    label=torch.tensor(sample_labels, dtype=torch.uint16),
                    bboxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]] * n).reshape(-1, 4),
                    masks=tv_tensors.Mask(masks),
                    dm_image_info=DmImageInfo(width=8, height=8),
                    subset=Subset.TRAINING,
                ),
            )
        dataset = InstanceSegDataset(dm_subset=dm_dataset, transforms=[])
        batch = _default_collate_fn([dataset[i] for i in range(len(labels))])
        assert batch.labels is not None
        assert all(label.dtype == torch.long for label in batch.labels)
        assert [label.tolist() for label in batch.labels] == labels
        assert dataset.get_idx_list_per_classes() == {0: [0], 255: [0], 256: [1], num_classes - 1: [1]}
