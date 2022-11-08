"""OpenVINO Training Extensions."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__version__ = "1.0.0"

import os
import tempfile


class OTXConstants:
    PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    CONFIG_PATH = os.path.join(PACKAGE_ROOT, 'configs')
    TEMP_PATH = tempfile.mkdtemp(prefix="otx-")
    # SAMPLES_PATH = os.path.join(PACKAGE_ROOT, 'samples')
    # MODELS_PATH = os.path.join(PACKAGE_ROOT, 'models')


__all__ = [
    "OTXConstants"
]
