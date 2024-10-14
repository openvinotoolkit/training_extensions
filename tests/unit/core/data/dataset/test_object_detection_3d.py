# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests of Object Detection 3D datasets."""

from __future__ import annotations

import pytest
from datumaro import Dataset as DmDataset
from otx.core.data.dataset.object_detection_3d import OTX3DObjectDetectionDataset
from otx.core.data.entity.base import ImageInfo
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms.v2 import Identity, Transform


class TestOTXObjectDetection3DDataset:
    @pytest.fixture()
    def fxt_dm_dataset(self) -> DmDataset:
        return DmDataset.import_from("tests/assets/kitti3d", format="kitti3d")

    @pytest.fixture()
    def fxt_tvt_transforms(self) -> Identity:
        return Identity()

    @pytest.mark.parametrize("subset", ["train", "val"])
    def test_get_item_impl_subset(
        self,
        fxt_dm_dataset,
        fxt_tvt_transforms: Transform,
        subset: str,
    ) -> None:
        dataset = OTX3DObjectDetectionDataset(
            fxt_dm_dataset.get_subset(subset).as_dataset(),
            fxt_tvt_transforms,
        )

        entity = dataset._get_item_impl(0)

        assert hasattr(entity, "image")
        assert isinstance(entity.image, Tensor)
        assert hasattr(entity, "img_info")
        assert isinstance(entity.img_info, ImageInfo)
        assert hasattr(entity, "calib_matrix")
        assert isinstance(entity.calib_matrix, Tensor)
        assert hasattr(entity, "boxes_3d")
        assert isinstance(entity.boxes_3d, Tensor)
        assert hasattr(entity, "boxes")
        assert isinstance(entity.boxes, tv_tensors.BoundingBoxes)
        assert hasattr(entity, "size_2d")
        assert isinstance(entity.boxes_3d, Tensor)
        assert hasattr(entity, "size_3d")
        assert isinstance(entity.boxes_3d, Tensor)
        assert hasattr(entity, "heading_angle")
        assert isinstance(entity.boxes_3d, Tensor)
        assert hasattr(entity, "depth")
        assert isinstance(entity.boxes_3d, Tensor)
        assert hasattr(entity, "original_kitti_format")
        assert (
            entity.original_kitti_format is None
            if subset == "train"
            else isinstance(entity.original_kitti_format, dict)
        )
