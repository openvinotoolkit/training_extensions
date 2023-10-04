# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from otx.v2.adapters.openvino.ops.object_detections import (
    DetectionOutputV0,
    PriorBoxClusteredV0,
    PriorBoxV0,
    ProposalV4,
    RegionYoloV0,
    ROIPoolingV0,
)


class TestProposalV4:

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid framework error."):
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


    def test_forward(self) -> None:
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

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid method error."):
            ROIPoolingV0(
                "dummy",
                shape=(1,),
                pooled_h=1,
                pooled_w=1,
                spatial_scale=1.0,
                method="error",
            )


    def test_forward(self) -> None:
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

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid code_type error."):
            DetectionOutputV0(
                "dummy",
                shape=(1,),
                keep_top_k=[1],
                nms_threshold=0.1,
                code_type="error",
            )


    def test_forward(self) -> None:
        op = DetectionOutputV0(
            "dummy",
            shape=(1,),
            keep_top_k=[1],
            nms_threshold=0.1,
        )
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy", "dummy")


class TestRegionYoloV0:

    def test_forward(self) -> None:
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

    def test_forward(self) -> None:
        op = PriorBoxV0("dummy", shape=(1,), offset=0.1)
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy")


class TestPriorBoxClusteredV0:

    def test_forward(self) -> None:
        op = PriorBoxClusteredV0("dummy", shape=(1,), offset=0.1)
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy")
