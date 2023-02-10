"""Unit Test for otx.algorithms.detection.adapters.mmdet.utils.builder."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from otx.algorithms.detection.adapters.mmdet.utils import build_detector
from otx.mpa.utils.config_utils import MPAConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_MODEL_CONFIG_PATH,
    DEFAULT_ISEG_MODEL_CONFIG_PATH,
)


class TestOTXDetBuilder:
    """Test OTXDetBuilder class."""

    @e2e_pytest_unit
    @pytest.mark.parametrize("model_cfg", [DEFAULT_DET_MODEL_CONFIG_PATH, DEFAULT_ISEG_MODEL_CONFIG_PATH])
    @pytest.mark.parametrize("cfg_options", [None, MPAConfig()])
    def test_build_detector(self, model_cfg, cfg_options):
        """Test build_detector method."""
        cfg = MPAConfig.fromfile(model_cfg)
        cfg.model.type = cfg.model.type.replace("Custom", "")
        cfg.model.backbone.type = cfg.model.backbone.type.replace("mobilenetv2_w1", "MobileNetV2")
        cfg.model.backbone.pretrained = None
        model = build_detector(cfg, checkpoint=cfg.load_from, cfg_options=cfg_options)
        assert model is not None
