# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from getitune.backend.openvino.models.utils import get_default_num_async_infer_requests


def test_get_default_num_async_infer_requests() -> None:
    # Test the get_default_num_async_infer_requests function.

    # Mock os.cpu_count() to return a specific value
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 4

    # Call the function and check the return value
    assert get_default_num_async_infer_requests() == 2

    # Restore the original os.cpu_count() function
    os.cpu_count = original_cpu_count

    # Check the warning message
    with pytest.warns(UserWarning, match="Set the default number of OpenVINO inference requests"):
        get_default_num_async_infer_requests()
