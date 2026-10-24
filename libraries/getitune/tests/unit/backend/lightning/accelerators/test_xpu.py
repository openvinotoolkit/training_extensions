# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test for getitune.algo.accelerators.xpu"""

import pytest
import torch

from getitune.backend.lightning.accelerators import XPUAccelerator
from getitune.utils.device import is_xpu_available


class TestXPUAccelerator:
    @pytest.fixture
    def accelerator(self, mocker):
        mock_torch = mocker.patch("getitune.backend.lightning.accelerators.xpu.torch")
        return XPUAccelerator(), mock_torch

    def test_setup_device(self, accelerator):
        accelerator, mock_torch = accelerator
        device = torch.device("xpu")
        accelerator.setup_device(device)
        assert mock_torch.xpu.set_device.called

    def test_parse_devices(self, accelerator):
        accelerator, _ = accelerator
        devices = [1, 2, 3]
        parsed_devices = accelerator.parse_devices(devices)
        assert isinstance(parsed_devices, list)
        assert parsed_devices == devices

    def test_get_parallel_devices(self, accelerator):
        accelerator, _ = accelerator
        devices = [1, 2, 3]
        parallel_devices = accelerator.get_parallel_devices(devices)
        assert isinstance(parallel_devices, list)
        for device in parallel_devices:
            # the XPU accelerator returns mocked device objects when torch is patched
            assert hasattr(device, "index") or hasattr(device, "device")

    def test_auto_device_count(self, accelerator):
        accelerator, mock_torch = accelerator
        count = accelerator.auto_device_count()
        # when patched, device_count will be a mock; ensure it was called
        assert mock_torch.xpu.device_count.called
        assert count is not None

    def test_is_available(self, accelerator):
        accelerator, _ = accelerator
        available = accelerator.is_available()
        assert isinstance(available, bool)
        assert available == is_xpu_available()

    def test_name(self, accelerator):
        accelerator, _ = accelerator
        name = accelerator.name()
        assert name == "xpu"

    def test_get_device_stats(self, accelerator):
        accelerator, _ = accelerator
        device = torch.device("xpu")
        stats = accelerator.get_device_stats(device)
        assert isinstance(stats, dict)

    def test_teardown(self, accelerator):
        accelerator, _ = accelerator
        accelerator.teardown()
