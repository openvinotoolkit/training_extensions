# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for physical benchmark hardware discovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from getitune.benchmark.hardware import get_openvino_device_name, get_training_device_name


def test_training_cuda_device_name(mocker) -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.current_device.return_value = 0
    torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3090"
    mocker.patch.dict("sys.modules", {"torch": torch})

    assert get_training_device_name("cuda") == "NVIDIA GeForce RTX 3090"


def test_training_xpu_device_name(mocker) -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.xpu.is_available.return_value = True
    torch.xpu.current_device.return_value = 0
    torch.xpu.get_device_name.return_value = "Intel(R) Arc(TM) B-Series Graphics"
    mocker.patch.dict("sys.modules", {"torch": torch})

    assert get_training_device_name("xpu") == "Intel(R) Arc(TM) B-Series Graphics"


def test_missing_training_device_raises(mocker) -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.xpu.is_available.return_value = False
    mocker.patch.dict("sys.modules", {"torch": torch})

    with pytest.raises(RuntimeError, match="physical training device"):
        get_training_device_name("cuda")


def test_openvino_full_device_name(mocker) -> None:
    core = MagicMock()
    core.available_devices = ["CPU", "GPU.0"]
    core.get_property.return_value = "Intel(R) Arc(TM) B-Series Graphics"
    mocker.patch("openvino.Core", return_value=core)

    assert get_openvino_device_name("GPU") == "Intel(R) Arc(TM) B-Series Graphics"
    core.get_property.assert_called_once_with("GPU.0", "FULL_DEVICE_NAME")
