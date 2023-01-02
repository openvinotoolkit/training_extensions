# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn
from mmcv.runner import HOOKS, Hook

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class NoBiasDecayHook(Hook):
    """Hook for No Bias Decay Method (Bag of Tricks for Image Classification)

    This hook divides model's weight & bias to 3 parameter groups
    [weight with decay, weight without decay, bias without decay]
    """

    def before_train_epoch(self, runner):
        weight_decay, bias_no_decay, weight_no_decay = [], [], []
        for m in runner.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_decay.append(m.weight)
                if m.bias is not None:
                    bias_no_decay.append(m.bias)
            elif hasattr(m, "weight") or hasattr(m, "bias"):
                if hasattr(m, "weight"):
                    weight_no_decay.append(m.weight)
                if hasattr(m, "bias"):
                    bias_no_decay.append(m.bias)
            elif len(list(m.children())) == 0:
                for p in m.parameters():
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
        params = []
        for m in runner.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                params.append(m.weight)
                if m.bias is not None:
                    params.append(m.bias)
            elif hasattr(m, "weight") or hasattr(m, "bias"):
                if hasattr(m, "weight"):
                    params.append(m.weight)
                if hasattr(m, "bias"):
                    params.append(m.bias)
            elif len(list(m.children())) == 0:
                for p in m.parameters():
                    params.append(p)

        param_groups = runner.optimizer.param_groups[0].copy()
        param_groups["params"] = params
        runner.optimizer.param_groups = [param_groups]
