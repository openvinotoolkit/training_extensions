# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.algo.action_classification.movinet import MoViNet
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestMoViNet:
    @pytest.fixture()
    def fxt_movinet(self) -> MoViNet:
        return MoViNet(label_info=10)

    def test_load_from_otx_v1_ckpt(self, fxt_movinet, mocker) -> None:
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_action_ckpt")
        fxt_movinet.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")
