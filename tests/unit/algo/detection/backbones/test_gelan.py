# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of GELAN architecture."""

import pytest
import torch
from otx.algo.detection.backbones.gelan import GELAN, GELANModule
from otx.algo.detection.utils.utils import auto_pad
from otx.algo.modules import Conv2dModule
from torch import nn


class TestGELAN:
    """Test GELANModule."""

    @pytest.fixture()
    def gelan(self) -> GELANModule:
        return GELAN(model_name="yolov9_s")

    def test_init(self, gelan: GELANModule) -> None:
        cfg = GELAN.GELAN_CFG["yolov9_s"]

        assert gelan.first_dim == cfg["first_dim"]
        assert gelan.block_entry_cfg == cfg["block_entry_cfg"]
        assert gelan.csp_channels == cfg["csp_channels"]
        assert gelan.csp_args == cfg["csp_args"]

        assert len(gelan.module) == 9
        assert isinstance(gelan.module[0], Conv2dModule)
        assert gelan.module[0].in_channels == 3
        assert gelan.module[0].out_channels == cfg["first_dim"]
        assert gelan.module[0].kernel_size == (3, 3)
        assert gelan.module[0].stride == (2, 2)
        assert gelan.module[0].padding == auto_pad(kernel_size=3)
        assert isinstance(gelan.module[0].bn, nn.BatchNorm2d)
        assert isinstance(gelan.module[0].activation, nn.SiLU)

    def test_forward(self, gelan: GELANModule) -> None:
        x = torch.randn(1, 3, 640, 640)
        results = gelan(x)

        assert list(results.keys()) == [0, -1, "B3", "B4", "B5"]
        assert results[0].shape == (1, 3, 640, 640)
        assert results["B3"].shape == (1, 128, 80, 80)
        assert results["B4"].shape == (1, 192, 40, 40)
        assert results["B5"].shape == (1, 256, 20, 20)
        assert results[-1].shape == (1, 256, 20, 20)
