# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest

from otx.core.ov.ops.object_detections import (
    DetectionOutputV0,
    PriorBoxClusteredV0,
    PriorBoxV0,
    ProposalV4,
    RegionYoloV0,
    ROIPoolingV0,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestProposalV4:
    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            ProposalV4(
                "dummy",
                shape=(1,),
                base_size=1,
                pre_nms_topn=1,
                post_nms_topn=1,
                nms_thresh=0.1,
                feat_stride=1,
                min_size=1,
                ratio=[1.0],
                scale=[1.0],
                framework="error",
            )

    @e2e_pytest_unit
    def test_forward(self):
        op = ProposalV4(
            "dummy",
            shape=(1,),
            base_size=1,
            pre_nms_topn=1,
            post_nms_topn=1,
            nms_thresh=0.1,
            feat_stride=1,
            min_size=1,
            ratio=[1.0],
            scale=[1.0],
        )
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy", "dummy")


class TestROIPoolingV0:
    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            ROIPoolingV0(
                "dummy",
                shape=(1,),
                pooled_h=1,
                pooled_w=1,
                spatial_scale=1.0,
                method="error",
            )

    @e2e_pytest_unit
    def test_forward(self):
        op = ROIPoolingV0(
            "dummy",
            shape=(1,),
            pooled_h=1,
            pooled_w=1,
            spatial_scale=1.0,
        )
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy")


class TestDetectionOutputV0:
    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            DetectionOutputV0(
                "dummy",
                shape=(1,),
                keep_top_k=[1],
                nms_threshold=0.1,
                code_type="error",
            )

    @e2e_pytest_unit
    def test_forward(self):
        op = DetectionOutputV0(
            "dummy",
            shape=(1,),
            keep_top_k=[1],
            nms_threshold=0.1,
        )
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy", "dummy")


class TestRegionYoloV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = RegionYoloV0(
            "dummy",
            shape=(1,),
            axis=1,
            coords=1,
            classes=1,
            end_axis=1,
            num=1,
        )
        with pytest.raises(NotImplementedError):
            op("dummy")


class TestPriorBoxV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = PriorBoxV0("dummy", shape=(1,), offset=0.1)
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy")


class TestPriorBoxClusteredV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = PriorBoxClusteredV0("dummy", shape=(1,), offset=0.1)
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy")
