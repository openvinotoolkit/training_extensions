# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test Base Dataset of OTX."""

from __future__ import annotations

import numpy as np
import pytest
from datumaro.components.media import ImageFromFile
from otx.core.data.dataset.base import OTXDataset
from otx.core.data.mem_cache import MemCacheHandlerBase


class TestOTXDataset:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class MockItem:
            def __init__(self, idx) -> None:
                self.id = idx

        def mock_transform(inputs: int) -> int:
            return inputs + 1

        self.mock_dm_subset = [MockItem(0), MockItem(1), MockItem(2), MockItem(3)]
        self.mock_transforms = [mock_transform, mock_transform, mock_transform]
        self.dataset = OTXDataset(self.mock_dm_subset, mock_transform, (512, 512), 10)

    def test_sample_another_idx(self) -> None:
        assert self.dataset._sample_another_idx() < len(self.dataset)

    def test_apply_transforms(self) -> None:
        mock_entity = 0
        assert self.dataset._apply_transforms(mock_entity) == 1

        self.dataset = OTXDataset(self.mock_dm_subset, self.mock_transforms, (512, 512), 10)
        assert self.dataset._apply_transforms(mock_entity) == 3

    def test_getitem(self, mocker) -> None:
        mocker.patch.object(OTXDataset, "_get_item_impl", return_value=1)
        assert self.dataset[0] == 1

        mocker.patch.object(OTXDataset, "_get_item_impl", return_value=None)
        with pytest.raises(RuntimeError):
            self.dataset[0]

    def test_get_img_data_and_shape(self) -> None:
        class MockImageFromFile(ImageFromFile):
            def __init__(self, path, data):
                super().__init__(path=path)
                self._data = data

            @property
            def data(self) -> np.ndarray:
                return self._data

        random_genertor = np.random.default_rng(1)
        mock_data = random_genertor.integers(0, 256, size=(224, 224)).astype(np.float32)
        img_data, shape = self.dataset._get_img_data_and_shape(MockImageFromFile("temp", mock_data))
        assert shape == (224, 224)
        assert img_data.shape == (224, 224, 3)

    def test_cache_img(self, mocker) -> None:
        class MockMemCacheHandlerBase(MemCacheHandlerBase):
            def __init__(self, dsize: tuple) -> None:
                self.dsize = dsize

            def put(self, key: str | int, data: np.ndarray, meta: dict | None = None) -> None:
                """Check whether data shape is same as we expected."""
                assert data.shape == self.dsize

        self.dataset._cache_img(MockMemCacheHandlerBase((224, 224, 3)), "key", np.ndarray((224, 224, 3)), (480, 480))
        self.dataset._cache_img(MockMemCacheHandlerBase((512, 512, 3)), "key", np.ndarray((224, 224, 3)), (640, 640))


        self.dataset = OTXDataset(self.mock_dm_subset, self.mock_transforms[0], None, 10)
        self.dataset._cache_img(MockMemCacheHandlerBase((224, 224, 3)), "key", np.ndarray((224, 224, 3)), (640, 640))
