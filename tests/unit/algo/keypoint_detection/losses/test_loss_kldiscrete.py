# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of KLDiscreteLoss."""

import torch
from otx.algo.keypoint_detection.losses.kl_discret_loss import KLDiscretLoss


class TestKLDiscretLoss:
    def test_forward(self) -> None:
        """Test forward."""
        batch_size = 2
        n_kpts = 7
        x_dim, y_dim = 384, 512

        pred_sim_cc = (torch.rand(batch_size, n_kpts, x_dim), torch.rand(batch_size, n_kpts, y_dim))
        gt_sim_cc = (torch.rand(batch_size, n_kpts, x_dim), torch.rand(batch_size, n_kpts, y_dim))
        keypoint_weights = torch.rand(batch_size, n_kpts)

        loss_module = KLDiscretLoss()
        loss = loss_module.forward(pred_sim_cc, gt_sim_cc, keypoint_weights)
        assert loss.shape == torch.Size(())

        loss_module = KLDiscretLoss(label_softmax=True)
        loss = loss_module.forward(pred_sim_cc, gt_sim_cc, keypoint_weights)
        assert loss.shape == torch.Size(())
