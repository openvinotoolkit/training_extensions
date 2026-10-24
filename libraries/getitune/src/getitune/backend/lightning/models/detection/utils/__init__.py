# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utils for detection task."""

from .rfdetr_batch_utils import limit_batch_objects
from .utils import generate_scales

__all__ = ["generate_scales", "limit_batch_objects"]
