# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from otx.algo.action_classification.x3d import X3D
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestX3D:
    @pytest.fixture()
    def fxt_x3d(self, fxt_multiclass_labelinfo) -> X3D:
        return X3D(label_info=10)

    def test_load_from_otx_v1_ckpt(self, fxt_x3d, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_action_ckpt")
        fxt_x3d.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")
