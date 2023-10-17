"""Dual model EMA hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner

from otx.v2.api.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class DualModelEMAHook(Hook):
    r"""Generalized re-implementation of mmengine.hooks.EMAHook.

    Source model paramters would be exponentially averaged
    onto destination model pararmeters on given intervals

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        epoch_momentum (float): if > 0, momentum is ignored and re-calculated.
            momentum = 1 - exp(1 - epoch_momentum, 1/num_iter_per_epoch)
            Defaults to 0.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        start_epoch (int): During initial a few epochs, we just copy values
            to update ema parameters. Defaults to 5.
        src_model_name (str): Source model for EMA (X)
        dst_model_name (str): Destination model for EMA (Xema)
    """

    def __init__(
        self,
        momentum: float = 0.0002,
        epoch_momentum: float = 0.0,
        interval: float = 1,
        start_epoch: int = 5,
        src_model_name: str = "model_s",
        dst_model_name: str = "model_t",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.momentum = 1 - (1 - momentum) ** interval
        self.epoch_momentum = epoch_momentum
        self.interval = interval
        self.start_epoch = start_epoch
        self.src_model = None
        self.dst_model = None
        self.src_model_name = src_model_name
        self.dst_model_name = dst_model_name
        self.src_params: dict = {}
        self.dst_params: dict = {}
        self.enabled = False

    def before_run(self, runner: Runner) -> None:
        """Set up src & dst model parameters."""
        model = self._get_model(runner)
        self.src_model = getattr(model, self.src_model_name, None)
        self.dst_model = getattr(model, self.dst_model_name, None)
        if self.src_model and self.dst_model:
            self.src_params = self.src_model.state_dict(keep_vars=True)
            self.dst_params = self.dst_model.state_dict(keep_vars=True)

    def before_train_epoch(self, runner: Runner) -> None:
        """Momentum update."""
        if runner.epoch + 1 == self.start_epoch:
            self._copy_model()
            self.enabled = True

        if self.epoch_momentum > 0.0 and self.enabled:
            iter_per_epoch = len(runner.data_loader)
            epoch_decay = 1 - self.epoch_momentum
            iter_decay = math.pow(epoch_decay, self.interval / iter_per_epoch)
            self.momentum = 1 - iter_decay
            logger.info(f"EMA: epoch_decay={epoch_decay} / iter_decay={iter_decay}")
            self.epoch_momentum = 0.0  # disable re-compute

    def after_train_iter(self, runner: Runner) -> None:
        """Update ema parameter every self.interval iterations."""
        if not self.enabled or (runner.iter % self.interval != 0):
            return

        # EMA
        self._ema_model()

    def after_train_epoch(self, runner: Runner) -> None:
        """Log difference between models if enabled."""
        if self.enabled:
            logger.info(f"model_s model_t diff: {self._diff_model()}")

    def _get_model(self, runner: Runner) -> torch.nn.Module:
        model = runner.model
        if hasattr(model, "module"):
            model = model.module
        return model

    def _copy_model(self) -> None:
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                if not name.startswith("ema_"):
                    dst_param = self.dst_params[name]
                    dst_param.data.copy_(src_param.data)

    def _ema_model(self) -> None:
        momentum = min(self.momentum, 1.0)
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                if not name.startswith("ema_"):
                    dst_param = self.dst_params[name]
                    dst_param.data.copy_(dst_param.data * (1 - momentum) + src_param.data * momentum)

    def _diff_model(self) -> float:
        diff_sum = 0.0
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                if not name.startswith("ema_"):
                    dst_param = self.dst_params[name]
                    diff = ((src_param - dst_param) ** 2).sum()
                    diff_sum += diff
        return diff_sum
