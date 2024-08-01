# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for build function."""

from __future__ import annotations

import warnings


def get_default_num_async_infer_requests() -> int:
    """Returns a default number of infer request for OV models."""
    import os

    number_requests = os.cpu_count()
    number_requests = max(1, int(number_requests / 2)) if number_requests is not None else 1
    msg = f"""Set the default number of OpenVINO inference requests to {number_requests}.
            You can specify the value in config."""
    warnings.warn(msg, stacklevel=1)
    return number_requests
