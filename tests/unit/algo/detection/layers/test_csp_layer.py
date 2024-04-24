# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of CSPLayer."""

from otx.algo.detection.layers import CSPLayer


class TestCSPLayer:
    def test_init(self) -> None:
        csp_layer = CSPLayer(3, 5)
