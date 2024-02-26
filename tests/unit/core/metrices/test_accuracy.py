# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of Module for OTX custom metrices."""

import torch
from otx.core.metrices.accuracy import (
    CustomMulticlassAccuracy,
    CustomMultilabelAccuracy,
    CustomHlabelAccuracy
)


class TestAccuracy:
    def test_fmeasure(self) -> None:
        """Check whether accuracy is same with OTX1.x version."""