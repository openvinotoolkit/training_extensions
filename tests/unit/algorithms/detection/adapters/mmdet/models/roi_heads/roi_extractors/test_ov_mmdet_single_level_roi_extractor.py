# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.detection.adapters.mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import (
    RoIInterpolationPool,
    SingleRoIExtractor,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSingleRoIExtractor:
    @e2e_pytest_unit
    def test_build_roi_layers(self):
        extractor = SingleRoIExtractor(
            roi_layer=dict(type="RoIInterpolationPool", output_size=14, mode="bilinear"),
            out_channels=1024,
            featmap_strides=[8],
        )
        assert all(isinstance(layer, RoIInterpolationPool) for layer in extractor.roi_layers)


#  class TestRoIInterpolationPool:
#      @pytest.fixture(autouse=True)
#      def setup(self):
#          self.pool = RoIInterpolationPool(14, 1/8)
#
#      @e2e_pytest_unit
#      def test_forward(self):
#          self.pool()
