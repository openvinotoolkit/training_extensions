# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom 3D recognizers for OTX."""

from .movinet_recognizer import MoViNetRecognizer
from .recognizer import OTXRecognizer3D

__all__ = ["OTXRecognizer3D", "MoViNetRecognizer"]
