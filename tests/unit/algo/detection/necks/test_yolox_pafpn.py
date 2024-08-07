# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of YOLOXPAFPN.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_necks/test_necks.py#L360-L387
"""

import torch
from otx.algo.detection.necks.yolox_pafpn import YOLOXPAFPN
from otx.algo.modules.conv_module import DepthwiseSeparableConvModule


class TestYOLOXPAFPN:
    def test_yolox_pafpn(self) -> None:
        s = 64
        in_channels = [8, 16, 32, 64]
        feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
        out_channels = 24
        feats = [torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i]) for i in range(len(in_channels))]
        neck = YOLOXPAFPN(in_channels=in_channels, out_channels=out_channels)
        outs = neck(feats)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

        # test depth-wise
        neck = YOLOXPAFPN(in_channels=in_channels, out_channels=out_channels, use_depthwise=True)

        assert isinstance(neck.downsamples[0], DepthwiseSeparableConvModule)

        outs = neck(feats)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
