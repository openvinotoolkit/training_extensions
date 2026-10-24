# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests the XPU strategy."""

import pytest
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from getitune.backend.lightning.strategies import xpu_single as target_file
from getitune.backend.lightning.strategies.xpu_single import SingleXPUStrategy


class TestSingleXPUStrategy:
    @pytest.fixture
    def mock_is_xpu_available(self, mocker):
        return mocker.patch.object(target_file, "is_xpu_available", return_value=True)

    def test_init(self, mock_is_xpu_available):
        strategy = SingleXPUStrategy(device="xpu:0")
        assert mock_is_xpu_available.call_count == 1
        assert strategy._root_device.type == "xpu"
        assert strategy.accelerator is None

    def test_init_no_xpu(self, mock_is_xpu_available):
        mock_is_xpu_available.return_value = False
        with pytest.raises(MisconfigurationException):
            SingleXPUStrategy(device="xpu:0")

    @pytest.fixture
    def strategy(self, mock_is_xpu_available):
        return SingleXPUStrategy(device="xpu:0", accelerator="xpu")
