"""Module for NoBiasDecayHook used in classification."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook
from torch import nn

from otx.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class NoBiasDecayHook(Hook):
    """Hook for No Bias Decay Method (Bag of Tricks for Image Classification).

    This hook divides model's weight & bias to 3 parameter groups
    [weight with decay, weight without decay, bias without decay].
    """

    def before_train_epoch(self, runner):
        """Split weights into decay/no-decay groups."""
        weight_decay, bias_no_decay, weight_no_decay = [], [], []
        for module in runner.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_decay.append(module.weight)
                if module.bias is not None:
                    bias_no_decay.append(module.bias)
            elif hasattr(module, "weight") or hasattr(module, "bias"):
                if hasattr(module, "weight"):
                    weight_no_decay.append(module.weight)
                if hasattr(module, "bias"):
                    bias_no_decay.append(module.bias)
            elif len(list(module.children())) == 0:
                for p in module.parameters():
                    weight_decay.append(p)

        weight_decay_group = runner.optimizer.param_groups[0].copy()
        weight_decay_group["params"] = weight_decay

        bias_group = runner.optimizer.param_groups[0].copy()
        bias_group["params"] = bias_no_decay
        bias_group["weight_decay"] = 0.0
        bias_group["lr"] *= 2

        weight_no_decay_group = runner.optimizer.param_groups[0].copy()
        weight_no_decay_group["params"] = weight_no_decay
        weight_no_decay_group["weight_decay"] = 0.0

        param_groups = [weight_decay_group, bias_group, weight_no_decay_group]
        runner.optimizer.param_groups = param_groups

    def after_train_epoch(self, runner):
        """Merge splited groups before saving checkpoint."""
        params = []
        for module in runner.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params.append(module.weight)
                if module.bias is not None:
                    params.append(module.bias)
            elif hasattr(module, "weight") or hasattr(module, "bias"):
                if hasattr(module, "weight"):
                    params.append(module.weight)
                if hasattr(module, "bias"):
                    params.append(module.bias)
            elif len(list(module.children())) == 0:
                for p in module.parameters():
                    params.append(p)

        param_groups = runner.optimizer.param_groups[0].copy()
        param_groups["params"] = params
        runner.optimizer.param_groups = [param_groups]
