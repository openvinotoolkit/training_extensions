"""Test for CustomDeformableDETR Detector."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import build_detector

from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_deformable_detr_detector import (
    CustomDeformableDETR,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomDeformableDETR:
    @e2e_pytest_unit
    def test_custom_deformable_detr_build(self, fxt_cfg_custom_deformable_detr):
        model = build_detector(fxt_cfg_custom_deformable_detr)
        assert isinstance(model, CustomDeformableDETR)
        assert model.task_adapt is not None
