# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test detection dataset."""

import numpy as np
import torch
from datumaro import Bbox
from otx.core.data.dataset.detection import OTXDetectionDataset
from otx.core.data.entity.detection import DetDataEntity


class TestOTXDetectionDataset:
    def test_get_item_impl(self, mocker) -> None:
        class MockImage:
            size = (224, 224, 3)

        class MockItem:
            def __init__(self, idx: int):
                self.id = idx
                self.annotations = [Bbox(100, 100, 150, 150, label=0), Bbox(50, 50, 100, 100, label=1)]

            def media_as(self, return_type) -> MockImage:
                return MockImage()

        class MockDMSubset:
            def __init__(self, items) -> None:
                self.items = items
                self.name = "train"

            def __getitem__(self, index: int) -> MockItem:
                return self.items[index]

            def get(self, id: int, subset: str) -> MockItem:  # noqa: A002
                return self[id]

        def mock_transform(inputs: DetDataEntity) -> DetDataEntity:
            return inputs

        random_genertor = np.random.default_rng(1)
        mock_img = random_genertor.integers(0, 256, size=(224, 224, 3)).astype(np.float32)
        mocker.patch.object(OTXDetectionDataset, "_get_img_data", return_value=mock_img)

        self.dataset = OTXDetectionDataset(MockDMSubset([MockItem(0), MockItem(1)]), mock_transform)
        out = self.dataset._get_item_impl(0)
        assert out.image.shape == (224, 224, 3)
        assert torch.all(out.bboxes[0] == torch.Tensor([100., 100., 250., 250.]))
        assert torch.all(out.bboxes[1] == torch.Tensor([50., 50., 150., 150.]))
        assert out.img_info.scale_factor == (1.0, 1.0)
