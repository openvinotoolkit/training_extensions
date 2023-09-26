"""Quantization preset Enums for post training optimization."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional, Tuple

from otx.api.configuration import ConfigurableEnum


class POTQuantizationPreset(ConfigurableEnum):
    """This Enum represents the quantization preset for post training optimization."""

    PERFORMANCE = "Performance"
    MIXED = "Mixed"


class StorageCacheScheme(ConfigurableEnum):
    """This Enum represents the storage scheme for Datumaro arrow format."""

    NONE = "NONE"
    AS_IS = "AS-IS"
    JPEG_75 = "JPEG/75"
    JPEG_95 = "JPEG/95"
    PNG = "PNG"
    TIFF = "TIFF"


class BatchSizeAdaptType(ConfigurableEnum):
    """This Enum represents the type of adapting batch size.

    None : Not adapt batch size.
    Safe : Find a batch size preventing GPU out of memory.
    Full : Find a batch size using almost GPU memory.
    """

    NONE = "None"
    SAFE = "Safe"
    FULL = "Full"


class InputSizePreset(ConfigurableEnum):
    """Configurable input size preset."""

    DEFAULT = "Default"
    AUTO = "Auto"
    _64x64 = "64x64"
    _128x128 = "128x128"
    _224x224 = "224x224"
    _256x256 = "256x256"
    _384x384 = "384x384"
    _512x512 = "512x512"
    _640x640 = "640x640"
    _768x768 = "768x768"
    _1024x1024 = "1024x1024"

    @staticmethod
    def parse(value: str) -> Optional[Tuple[int, int]]:
        """Parse string value to tuple."""
        if value == "Default":
            return None
        if value == "Auto":
            return (0, 0)
        parsed_tocken = re.match("(\\d+)x(\\d+)", value)
        if parsed_tocken is None:
            return None
        return (int(parsed_tocken.group(1)), int(parsed_tocken.group(2)))

    @property
    def tuple(self) -> Optional[Tuple[int, int]]:
        """Returns parsed tuple."""
        return InputSizePreset.parse(self.value)

    @classmethod
    def input_sizes(cls):
        """Returns list of actual size tuples."""
        return [e.tuple for e in cls if e.value[0].isdigit()]
