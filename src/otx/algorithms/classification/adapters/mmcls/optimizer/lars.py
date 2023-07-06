"""Module for defining LARS optimizer for classification task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.runner import OPTIMIZERS
from torch.optim.optimizer import Optimizer, required


@OPTIMIZERS.register_module()
class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        dampening (float, optional): dampening for momentum (default: 0)
        eta (float, optional): LARS coefficient
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9,
        >>>                  weight_decay=1e-4, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        eta=0.001,
        nesterov=False,
        mode=None,
        exclude_bn_from_weight_decay=False,
    ):  # pylint: disable=too-many-arguments, too-many-locals
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eta < 0.0:
            raise ValueError(f"Invalid LARS coefficient value: {eta}")

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, eta=eta
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # split param group into weight_decay/non-weight decay.
        if exclude_bn_from_weight_decay:
            param_groups = list(params)
            if not isinstance(param_groups[0], dict):
                param_groups = [{"params": param_groups}]

            new_param_groups = []
            for param_group in param_groups:
                decay, no_decay = [], []
                for param in param_group.pop("params", []):
                    if not param.requires_grad:
                        continue

                    if len(param.shape) == 1:
                        no_decay.append(param)
                    else:
                        decay.append(param)

                decay_param_group = param_group.copy()
                decay_param_group["params"] = decay

                no_decay_param_group = param_group.copy()
                no_decay_param_group["params"] = no_decay
                no_decay_param_group["weight_decay"] = 0
                no_decay_param_group["lars_exclude"] = True

                new_param_groups.append(decay_param_group)
                new_param_groups.append(no_decay_param_group)

        self.mode = mode

        super().__init__(new_param_groups, defaults)

    def __setstate__(self, state):
        """Set state for parameter groups."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            eta = group["eta"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Add weight decay before computing adaptive LR.
                # Seems to be pretty important in SIMclr style models.
                local_lr = 1.0
                if not group.get("lars_exclude", False):
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(d_p).item()
                if self.mode == "selfsl" and weight_norm > 0 and grad_norm > 0:
                    local_lr = eta * weight_norm / grad_norm
                else:
                    local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                d_p = d_p.mul(local_lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - group["dampening"])
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf
                p.add_(d_p, alpha=-group["lr"])
        return loss
