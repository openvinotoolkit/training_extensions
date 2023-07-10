"""Quantization preset Enums for post training optimization."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

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
