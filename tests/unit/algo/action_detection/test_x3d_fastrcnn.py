# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from otx.algo.action_detection.x3d_fastrcnn import X3DFastRCNN
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestX3DFastRCNN:
    @pytest.fixture()
    def fxt_x3d_fast_rcnn(self) -> X3DFastRCNN:
        return X3DFastRCNN(num_classes=10, topk=3)

    def test_load_from_otx_v1_ckpt(self, fxt_x3d_fast_rcnn, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_action_ckpt")
        fxt_x3d_fast_rcnn.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")
