# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore


class TestAsymmetricAngularLoss:
    @pytest.fixture()
    def fxt_num_classes(self) -> None:
        return 2

    @pytest.fixture()
    def fxt_gt(self, fxt_num_classes) -> None:
        gt = torch.zeros((2, fxt_num_classes))
        gt[0, 0] = 1
        return gt

    @pytest.fixture()
    def fxt_input(self, fxt_num_classes) -> None:
        inputs = torch.zeros((2, fxt_num_classes))
        inputs[0, 0] = 1
        return inputs

    @pytest.fixture()
    def loss(self) -> None:
        return AsymmetricAngularLossWithIgnore(reduction="mean")

    def test_forward(self, loss, fxt_input, fxt_gt) -> None:
        result_c = loss(fxt_input, fxt_gt)
        fxt_input[0, 1] = 1
        result_w = loss(fxt_input, fxt_gt)
        assert result_c < result_w
