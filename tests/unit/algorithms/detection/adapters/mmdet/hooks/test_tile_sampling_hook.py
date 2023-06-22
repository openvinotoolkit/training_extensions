"""Test tiling sampling hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random

from otx.algorithms.detection.adapters.mmdet.hooks.tile_sampling_hook import TileSamplingHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestTilingSamplingHook:
    """Test class for TileSamplingHook."""

    @e2e_pytest_unit
    def test_before_epoch(self, mocker):
        "Test function for before_poch function."

        class MockTileDataset:
            def __init__(self):
                self.tiles_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                self.sample_num = 4
                self.tiles = [1, 2, 3, 4]

        class MockDataset:
            def __init__(self, tile_dataset):
                self.tile_dataset = tile_dataset

        class MockDataLoader:
            def __init__(self, dataset):
                self.dataset = dataset

        class MockRunner:
            def __init__(self, data_loader):
                self.data_loader = data_loader

        hook = TileSamplingHook()
        tile_dataset = MockTileDataset()
        runner = MockRunner(MockDataLoader(MockDataset(tile_dataset)))
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.hooks.tile_sampling_hook.sample", return_value=[5, 6, 7, 8]
        )
        hook.before_epoch(runner)
        assert tile_dataset.tiles[0] == 5
        assert tile_dataset.tiles[1] == 6
        assert tile_dataset.tiles[2] == 7
        assert tile_dataset.tiles[3] == 8
