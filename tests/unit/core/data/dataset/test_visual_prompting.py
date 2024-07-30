# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of visual prompting datasets."""

from __future__ import annotations

import numpy as np
import pytest
from datumaro import Dataset as DmDataset
from otx.core.data.dataset.visual_prompting import OTXVisualPromptingDataset, OTXZeroShotVisualPromptingDataset
from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.visual_prompting import ZeroShotVisualPromptingLabel
from torchvision.transforms.v2 import Identity, Transform
from torchvision.tv_tensors import BoundingBoxes, Image, Mask


class TestOTXVisualPromptingDataset:
    @pytest.fixture()
    def fxt_dm_dataset(self) -> DmDataset:
        return DmDataset.import_from("tests/assets/car_tree_bug", format="coco_instances")

    @pytest.fixture()
    def fxt_tvt_transforms(self, mocker) -> Identity:
        return Identity()

    @pytest.mark.parametrize("subset", ["train", "val"])
    @pytest.mark.parametrize("use_bbox", [True, False])
    @pytest.mark.parametrize("use_point", [True, False])
    def test_get_item_impl_subset(
        self,
        fxt_dm_dataset,
        fxt_tvt_transforms: Transform,
        subset: str,
        use_bbox: bool,
        use_point: bool,
    ) -> None:
        dataset = OTXVisualPromptingDataset(
            fxt_dm_dataset.get_subset(subset).as_dataset(),
            fxt_tvt_transforms,
            use_bbox=use_bbox,
            use_point=use_point,
        )

        if not use_bbox and not use_point:
            assert dataset.prob == 1.0

        entity = dataset._get_item_impl(0)

        assert hasattr(entity, "image")
        assert isinstance(entity.image, np.ndarray)
        assert hasattr(entity, "img_info")
        assert isinstance(entity.img_info, ImageInfo)
        assert hasattr(entity, "masks")
        assert isinstance(entity.masks, Mask)
        assert hasattr(entity, "labels")
        assert isinstance(entity.labels, dict)
        assert hasattr(entity, "polygons")
        assert isinstance(entity.polygons, list)
        assert hasattr(entity, "bboxes")
        assert hasattr(entity, "points")

        if not use_point:
            assert isinstance(entity.bboxes, BoundingBoxes)

        if not use_bbox and use_point:
            assert isinstance(entity.points, Points)


class TestOTXZeroShotVisualPromptingDataset:
    @pytest.fixture()
    def fxt_dm_dataset(self) -> DmDataset:
        return DmDataset.import_from("tests/assets/car_tree_bug", format="coco_instances")

    @pytest.fixture()
    def fxt_tvt_transforms(self, mocker) -> Identity:
        return Identity()

    @pytest.mark.parametrize("use_bbox", [True, False])
    @pytest.mark.parametrize("use_point", [True, False])
    def test_get_item_impl_subset(
        self,
        fxt_dm_dataset,
        fxt_tvt_transforms: Transform,
        use_bbox: bool,
        use_point: bool,
    ) -> None:
        dataset = OTXZeroShotVisualPromptingDataset(
            fxt_dm_dataset.get_subset("train").as_dataset(),
            fxt_tvt_transforms,
            use_bbox=use_bbox,
            use_point=use_point,
        )

        if not use_bbox and not use_point:
            assert dataset.prob == 1.0

        entity = dataset._get_item_impl(0)

        assert hasattr(entity, "image")
        assert isinstance(entity.image, Image)
        assert hasattr(entity, "img_info")
        assert isinstance(entity.img_info, ImageInfo)
        assert hasattr(entity, "masks")
        assert isinstance(entity.masks, Mask)
        assert hasattr(entity, "labels")
        assert isinstance(entity.labels, ZeroShotVisualPromptingLabel)
        assert hasattr(entity, "polygons")
        assert isinstance(entity.polygons, list)
        assert hasattr(entity, "prompts")

        if not use_point:
            assert all([isinstance(p, BoundingBoxes) for p in entity.prompts])  # noqa: C419

        if not use_bbox and use_point:
            assert all([isinstance(p, Points) for p in entity.prompts])  # noqa: C419
