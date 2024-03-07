"""Test for otx.algorithms.anomaly.adapters.anomalib.accelerators.xpu"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.algorithms.anomaly.adapters.anomalib.accelerators import XPUAccelerator
from otx.algorithms.common.utils import is_xpu_available


@pytest.mark.skipif(not is_xpu_available(), reason="XPU is not available")
class TestXPUAccelerator:
    @pytest.fixture
    def accelerator(self):
        return XPUAccelerator()

    def test_setup_device(self, accelerator):
        device = torch.device("xpu")
        accelerator.setup_device(device)

    def test_parse_devices(self, accelerator):
        devices = [1, 2, 3]
        parsed_devices = accelerator.parse_devices(devices)
        assert isinstance(parsed_devices, list)
        assert parsed_devices == devices

    def test_get_parallel_devices(self, accelerator):
        devices = [1, 2, 3]
        parallel_devices = accelerator.get_parallel_devices(devices)
        assert isinstance(parallel_devices, list)
        assert parallel_devices == [torch.device("xpu", idx) for idx in devices]

    def test_auto_device_count(self, accelerator):
        count = accelerator.auto_device_count()
        assert isinstance(count, int)

    def test_is_available(self, accelerator):
        available = accelerator.is_available()
        assert isinstance(available, bool)
        assert available == is_xpu_available()

    def test_get_device_stats(self, accelerator):
        device = torch.device("xpu")
        stats = accelerator.get_device_stats(device)
        assert isinstance(stats, dict)

    def test_teardown(self, accelerator):
        accelerator.teardown()
