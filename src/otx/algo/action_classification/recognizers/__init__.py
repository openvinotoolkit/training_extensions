# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom 3D recognizers for OTX."""

from .movinet_recognizer import MoViNetRecognizer
from .recognizer import BaseRecognizer

__all__ = ["BaseRecognizer", "MoViNetRecognizer"]
