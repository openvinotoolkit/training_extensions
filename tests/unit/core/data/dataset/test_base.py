# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test Base Dataset of OTX."""

import numpy as np
import pytest
from datumaro.components.media import ImageFromFile
from otx.core.data.dataset.base import OTXDataset


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
        self.dataset = OTXDataset(self.mock_dm_subset, mock_transform, 10)

    def test_sample_another_idx(self) -> None:
        assert self.dataset._sample_another_idx() < len(self.dataset)

    def test_apply_transforms(self) -> None:
        mock_entity = 0
        assert self.dataset._apply_transforms(mock_entity) == 1

        self.dataset = OTXDataset(self.mock_dm_subset, self.mock_transforms, 10)
        assert self.dataset._apply_transforms(mock_entity) == 3

    def test_get_img_data(self) -> None:
        class MockImageFromFile(ImageFromFile):
            def __init__(self, path, data):
                super().__init__(path=path)
                self._data = data

            @property
            def data(self) -> np.ndarray:
                return self._data

        random_genertor = np.random.default_rng(1)
        mock_data = random_genertor.integers(0, 256, size=(224, 224)).astype(np.float32)
        assert self.dataset._get_img_data(MockImageFromFile("temp", mock_data)).shape == (224, 224, 3)
