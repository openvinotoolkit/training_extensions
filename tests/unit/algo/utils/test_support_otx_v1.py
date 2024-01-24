# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import pytest
import torch
from otx.algo.utils.support_otx_v1 import OTXv1Helper

class TestOTXv1Helper:
    @pytest.fixture()
    def fxt_random_tensor(self) -> torch.Tensor:
        return torch.randn(3, 10)
    
    @pytest.mark.parametrize("label_type", ["multiclass", "multilabel", "hlabel"])    
    def test_load_cls_effnet_v2_ckpt(self, label_type: str, fxt_random_tensor:torch.Tensor):
        # Setup test data
        test_state_dict = {
            "model.classifier.some_key": fxt_random_tensor,
            "model.some_other_key": fxt_random_tensor
        }