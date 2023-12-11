# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of base data entity."""

from otx.core.data.entity.base import ImageType, OTXBatchDataEntity, OTXDataEntity


class TestOTXDataEntity:
    def test_image_type(
        self,
        fxt_numpy_data_entity,
        fxt_torchvision_data_entity,
    ) -> None:
        assert fxt_numpy_data_entity.image_type == ImageType.NUMPY
        assert fxt_torchvision_data_entity.image_type == ImageType.TV_IMAGE


class TestOTXBatchDataEntity:
    def test_collate_fn(self, mocker, fxt_torchvision_data_entity) -> None:
        mocker.patch.object(OTXDataEntity, "task", return_value="detection")
        mocker.patch.object(OTXBatchDataEntity, "task", return_value="detection")
        data_entities = [
            fxt_torchvision_data_entity,
            fxt_torchvision_data_entity,
            fxt_torchvision_data_entity,
        ]

        data_batch = OTXBatchDataEntity.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)
