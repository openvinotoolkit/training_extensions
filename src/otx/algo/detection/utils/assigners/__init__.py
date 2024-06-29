# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Assigners for detection task."""

from .atss_assigner import ATSSAssigner
from .sim_ota_assigner import SimOTAAssigner

__all__ = ["ATSSAssigner", "SimOTAAssigner"]
