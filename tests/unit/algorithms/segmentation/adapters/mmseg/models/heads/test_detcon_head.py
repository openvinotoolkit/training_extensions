# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy

import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg.models import DetConHead
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestDetConHead:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.detcon_head = DetConHead(
            predictor=dict(
                type="SelfSLMLP",
                in_channels=4,
                hid_channels=8,
                out_channels=4,
                norm_cfg=dict(type="BN1d", requires_grad=True),
                with_avg_pool=False,
            ),
            loss_cfg=dict(type="DetConLoss", temperature=0.1),
        )

    @e2e_pytest_unit
    def test_init(self):
        assert self.detcon_head.predictor.__class__.__name__ == "SelfSLMLP"
        assert self.detcon_head.detcon_loss.__class__.__name__ == "DetConLoss"

    @e2e_pytest_unit
    def test_init_weights(self):
        self.detcon_head.predictor.mlp[0].weight = torch.nn.Parameter(
            torch.ones_like(self.detcon_head.predictor.mlp[0].weight)
        )
        old_weights = deepcopy(self.detcon_head.predictor.mlp[0].weight)

        self.detcon_head.init_weights()

        new_weights = self.detcon_head.predictor.mlp[0].weight

        assert torch.all(old_weights != new_weights)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "projs,projs_tgt,ids,ids_tgt,batch_size,num_samples",
        [(torch.ones((4, 4)), torch.ones((4, 4)), torch.Tensor([0, 0, 0, 0]), torch.Tensor([1, 1, 1, 1]), 2, 1)],
    )
    def test_forward(
        self,
        projs: torch.Tensor,
        projs_tgt: torch.Tensor,
        ids: torch.Tensor,
        ids_tgt: torch.Tensor,
        batch_size: int,
        num_samples: int,
    ):
        loss = self.detcon_head(projs, projs_tgt, ids, ids_tgt, batch_size, num_samples)

        assert isinstance(loss, dict)
        assert "loss" in loss
