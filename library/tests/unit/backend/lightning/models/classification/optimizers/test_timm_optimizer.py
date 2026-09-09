# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from getitune.backend.lightning.models.classification.optimizers.timm import TimmOptimizer


class TestTimmOptimizer:
    @pytest.mark.parametrize(
        ("model_name", "expected_kwargs"),
        [
            ("vit_base_patch16_224", {"opt": "adamw", "weight_decay": 0.05}),
            ("vit_base_patch16_224.augreg_in21k", {"opt": "adamw", "weight_decay": 0.05}),  # dotted suffix
            ("VIT_base_patch16_224", {"opt": "adamw", "weight_decay": 0.05}),  # case-insensitive
            ("swin_tiny_patch4_window7_224", {"opt": "adamw", "weight_decay": 0.05}),
            ("tf_efficientnetv2_s.in21k", {"opt": "sgd", "momentum": 0.9, "weight_decay": 1e-4}),
            ("resnet50", {"opt": "sgd", "momentum": 0.9, "weight_decay": 1e-4}),
        ],
    )
    def test_selects_expected_optimizer_kwargs(self, model_name, expected_kwargs):
        optimizer_fn = TimmOptimizer(lr=0.01, weight_decay=float(expected_kwargs["weight_decay"]))
        optimizer_fn.bind_model_name(model_name)
        optimizer = optimizer_fn(nn.Linear(4, 2).parameters())

        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["weight_decay"] == expected_kwargs["weight_decay"]
        if expected_kwargs["opt"] == "sgd":
            assert optimizer.defaults["momentum"] == expected_kwargs["momentum"]
        else:
            assert isinstance(optimizer, torch.optim.AdamW)

    def test_produces_working_optimizer(self):
        optimizer = TimmOptimizer(lr=0.05, weight_decay=1e-4, model_name="resnet50")
        built = optimizer(nn.Linear(4, 2).parameters())
        assert isinstance(built, torch.optim.Optimizer)
        assert built.defaults["lr"] == 0.05

    def test_bind_model_name_sets_when_unset(self):
        optimizer = TimmOptimizer(lr=0.01, weight_decay=0.0)
        optimizer.bind_model_name("vit_base_patch16_224")
        assert optimizer.model_name == "vit_base_patch16_224"

    def test_call_without_bound_model_name_raises(self):
        optimizer = TimmOptimizer(lr=0.01, weight_decay=0.0)
        with pytest.raises(ValueError, match="model_name must be set"):
            optimizer(nn.Linear(4, 2).parameters())
