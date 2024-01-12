"""Config data type objects for device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from otx.core.types.device import DeviceType


@dataclass
class DeviceConfig:
    """Configuration class for the engine."""

    accelerator: DeviceType
    devices: int = 1
