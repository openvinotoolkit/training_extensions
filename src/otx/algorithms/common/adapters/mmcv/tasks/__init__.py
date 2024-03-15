"""Initialzation OTX Tasks with MMCV framework."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa
import os

from .version import __version__, get_version


class OTXConstants:
    """Various path for OTX."""

    PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # PACKAGE_ROOT = os.path.dirname(Path(__file__).)
    RECIPES_PATH = os.path.join(PACKAGE_ROOT, "recipes")
    SAMPLES_PATH = os.path.join(PACKAGE_ROOT, "samples")
    MODELS_PATH = os.path.join(PACKAGE_ROOT, "models")


# print(f'pkg root ======> {OTXConstants.PACKAGE_ROOT}')

__all__ = [
    "get_version",
    "__version__",
    "OTXConstants",
]
