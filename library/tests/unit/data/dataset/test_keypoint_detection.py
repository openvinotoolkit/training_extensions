# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of keypoint detection datasets."""

from __future__ import annotations

import polars as pl
import pytest
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import KeypointCategories, LabelCategories
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms.v2 import Identity, Transform

from getitune.data.dataset.keypoint_detection import KeypointDetectionDataset
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import KeypointSample


class TestKeypointDetectionDataset:
    @pytest.fixture
    def fxt_dm_dataset(self) -> Dataset:
        """Build a small keypoint detection dataset with the new DM API."""
        categories = {
            "label": LabelCategories(labels=("car", "tree", "bug", "person")),
            "keypoints": KeypointCategories(labels=("kp0", "kp1", "kp2", "kp3")),
        }
        ds = Dataset(KeypointSample, categories=categories)  # type: ignore[arg-type]

        num_keypoints = 4
        for i in range(4):
            h, w = 10, 10
            img = tv_tensors.Image(torch.zeros(3, h, w, dtype=torch.uint8))
            # Random keypoints: (num_keypoints, 3) with [x, y, visibility]
            kps = torch.zeros(num_keypoints, 3, dtype=torch.float32)
            kps[:, 0] = torch.arange(num_keypoints, dtype=torch.float32)  # x
            kps[:, 1] = torch.arange(num_keypoints, dtype=torch.float32)  # y
            kps[:, 2] = torch.tensor([0, 1, 2, 1], dtype=torch.float32)  # visibility (2 should be clamped)

            sample = KeypointSample(
                image=img,
                label=torch.tensor([i % 4], dtype=torch.long),
                keypoints=kps,
                dm_image_info=DmImageInfo(width=w, height=h),
                subset=Subset.TRAINING,
            )
            ds.append(sample)
        return ds

    @pytest.fixture
    def fxt_tvt_transforms(self) -> Identity:
        return Identity()

    def test_get_item_impl(
        self,
        fxt_dm_dataset,
        fxt_tvt_transforms: Transform,
    ) -> None:
        dataset = KeypointDetectionDataset(
            fxt_dm_dataset,
            transforms=fxt_tvt_transforms,
        )

        entity = dataset._get_item_impl(0)
        assert hasattr(entity, "image")
        assert isinstance(entity.image, Tensor)
        assert hasattr(entity, "img_info")
        assert isinstance(entity.img_info, ImageInfo)
        assert hasattr(entity, "label")
        assert isinstance(entity.label, Tensor)
        assert hasattr(entity, "keypoints")
        assert isinstance(entity.keypoints, Tensor)
        assert entity.keypoints.shape == (4, 3)
        # visibility channel should be clamped to 1 at max
        visibility = entity.keypoints[:, 2]
        assert visibility.max().item() <= 1


class TestKeypointDetectionDatasetLabelRange:
    """Test that keypoint labels are not limited to the UInt8 range."""

    def test_label_field_is_uint16(self) -> None:
        assert KeypointSample.label.dtype == pl.UInt16()  # type: ignore[attr-defined]

    def test_labels_above_255(self) -> None:
        num_classes = 300
        labels = [0, 255, 256, num_classes - 1]
        categories = {
            "label": LabelCategories(labels=tuple(f"class_{i}" for i in range(num_classes))),
            "keypoints": KeypointCategories(labels=("kp0", "kp1")),
        }
        dm_dataset = Dataset(KeypointSample, categories=categories)  # type: ignore[arg-type]
        for label in labels:
            dm_dataset.append(
                KeypointSample(
                    image=tv_tensors.Image(torch.zeros(3, 8, 8, dtype=torch.uint8)),
                    label=torch.tensor([label], dtype=torch.uint16),
                    keypoints=torch.ones(2, 3, dtype=torch.float32),
                    dm_image_info=DmImageInfo(width=8, height=8),
                    subset=Subset.TRAINING,
                ),
            )
        dataset = KeypointDetectionDataset(dm_dataset, transforms=Identity())
        assert [dataset[i].label.tolist() for i in range(len(labels))] == [[label] for label in labels]  # type: ignore[attr-defined]
