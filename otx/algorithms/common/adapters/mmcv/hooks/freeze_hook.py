"""Module for NoBiasDecayHook used in classification."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeHook(Hook):
    """Hook for No Bias Decay Method (Bag of Tricks for Image Classification).

    This hook divides model's weight & bias to 3 parameter groups
    [weight with decay, weight without decay, bias without decay].
    """

    def before_run(self, runner):
        """Split weights into decay/no-decay groups."""
        # Freeze the backbone
        for name, param in runner.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Confirm the backbone is frozen
        for name, param in runner.model.named_parameters():
            print(name, param.requires_grad)
