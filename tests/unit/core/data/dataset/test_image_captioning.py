# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import pytest
import torch
from datumaro import AnnotationType, Caption, DatasetItem, Image
from datumaro.components.dataset import Dataset as DmDataset
from otx.core.data.dataset.image_captioning import ImageCaptionDataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.image_captioning import ImageCaptionBatchDataEntity, ImageCaptionDataEntity
from torchvision import tv_tensors


@pytest.fixture()
def fxt_dm_caption_item() -> DatasetItem:
    np_img = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    np_img[:, :, 0] = 0  # Set 0 for B channel
    np_img[:, :, 1] = 1  # Set 1 for G channel
    np_img[:, :, 2] = 2  # Set 2 for R channel

    _, np_bytes = cv2.imencode(".png", np_img)
    media = Image.from_bytes(np_bytes.tobytes())
    media.path = ""

    return DatasetItem(
        id="item",
        subset="train",
        media=media,
        annotations=[
            Caption(caption="caption1"),
            Caption(caption="caption2"),
        ],
    )


@pytest.fixture()
def fxt_mock_dm_caption_subset(mocker, fxt_dm_caption_item):
    mock_dm_subset = mocker.MagicMock(spec=DmDataset)
    mock_dm_subset.__getitem__.return_value = fxt_dm_caption_item
    mock_dm_subset.__len__.return_value = 1
    mock_dm_subset.ann_types.return_value = [
        AnnotationType.caption,
    ]
    return mock_dm_subset


class TestImageCaptionDataset:
    @pytest.fixture()
    def dataset(self, mocker, fxt_mock_dm_caption_subset):
        dataset = ImageCaptionDataset(
            dm_subset=fxt_mock_dm_caption_subset,
            transforms=[lambda x: x],
            mem_cache_img_max_size=None,
            max_refetch=3,
        )
        dataset._get_img_data_and_shape = mocker.MagicMock(return_value=(mocker.MagicMock(), (224, 224)))
        dataset._apply_transforms = mocker.MagicMock(side_effect=lambda x: x)
        dataset.image_color_channel = 3
        return dataset

    def test_get_item_impl(self, dataset):
        entity = dataset._get_item_impl(0)
        assert isinstance(entity, ImageCaptionDataEntity)
        assert entity.img_info.img_idx == 0
        assert entity.img_info.img_shape == (224, 224)
        assert entity.captions == ["caption1", "caption2"]

    def test_collate_fn(self, dataset):
        collate_fn = dataset.collate_fn
        assert callable(collate_fn)
        data_entities = [
            ImageCaptionDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                captions=["caption1", "caption2"],
            ),
            ImageCaptionDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=1, img_shape=(224, 224), ori_shape=(224, 224)),
                captions=["caption3"],
            ),
        ]
        batch = collate_fn(data_entities)
        assert isinstance(batch, ImageCaptionBatchDataEntity)
        assert len(batch.imgs_info) == 2
        assert len(batch.captions) == 2
