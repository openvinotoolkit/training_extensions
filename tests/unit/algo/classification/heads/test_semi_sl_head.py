# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.classification.heads import SemiSLLinearClsHead


class TestSemiSLClsHead:
    @pytest.fixture()
    def fxt_semi_sl_head(self):
        """Semi-SL for Classification Head Settings."""
        return SemiSLLinearClsHead(
            num_classes=10,
            in_channels=10,
        )

    def test_build_type_error(self):
        """Verifies that SemiSLClsHead parameters check with TypeError."""
        with pytest.raises(TypeError):
            SemiSLLinearClsHead(
                num_classes=[1],
                in_channels=10,
            )

        with pytest.raises(TypeError):
            SemiSLLinearClsHead(
                num_classes=10,
                in_channels=[1],
            )

    def test_head_initialize(self, fxt_semi_sl_head):
        """Verifies that SemiSLClsHead parameters check with ValueError."""
        assert fxt_semi_sl_head.num_classes == 10
        assert fxt_semi_sl_head.use_dynamic_threshold
        assert fxt_semi_sl_head.min_threshold == 0.5
        assert fxt_semi_sl_head.num_pseudo_label == 0

    @pytest.fixture()
    def fxt_head_inputs(self):
        """Semi-SL for Classification Head Expected Inputs."""
        return {
            "feats": {
                "labeled": torch.rand(2, 10),
                "unlabeled_weak": torch.rand(4, 10),
                "unlabeled_strong": torch.rand(4, 10),
            },
            "labels": {
                "labeled": torch.tensor([6, 8]),
                "unlabeled_weak": torch.tensor([]),
                "unlabeled_strong": torch.tensor([]),
            },
        }

    def test_classwise_acc(self, fxt_semi_sl_head, fxt_head_inputs):
        """Verifies that SemiSLClsHead classwise_acc function works."""
        unlabeled_batch_size = fxt_head_inputs["feats"]["unlabeled_weak"].shape[0]
        logits, labels, label_u, mask = fxt_semi_sl_head.get_logits(**fxt_head_inputs)
        logits_x, logits_u = logits
        assert logits_x.shape == fxt_head_inputs["feats"]["labeled"].shape
        assert logits_u.shape == fxt_head_inputs["feats"]["unlabeled_strong"].shape
        assert labels.shape == fxt_head_inputs["labels"]["labeled"].shape
        assert fxt_semi_sl_head.classwise_acc.shape == (10,)
        assert mask.sum().item() < float(unlabeled_batch_size)

        # Unlabeled Threshold set to 0
        fxt_semi_sl_head.classwise_acc = torch.zeros(10)
        logits, labels, label_u, mask = fxt_semi_sl_head.get_logits(**fxt_head_inputs)
        logits_x, logits_u = logits
        assert logits_x.shape == fxt_head_inputs["feats"]["labeled"].shape
        assert logits_u.shape == fxt_head_inputs["feats"]["unlabeled_strong"].shape
        assert label_u.shape[0] == unlabeled_batch_size
        assert mask.shape[0] == unlabeled_batch_size
        assert mask.sum().item() == float(unlabeled_batch_size)
        assert fxt_semi_sl_head.num_pseudo_label == int(unlabeled_batch_size)
