"""Custom Optimizer Hook for mixed precision training on XPU."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Union

from mmcv.runner.hooks import HOOKS, Fp16OptimizerHook

from otx.algorithms.common.adapters.torch.amp import XPUGradScaler


@HOOKS.register_module()
class BFp16XPUOptimizerHook(Fp16OptimizerHook):
    """Custom Optimizer Hook for mixed & lower precision training on XPU."""

    def __init__(
        self,
        grad_clip: Optional[dict] = None,
        coalesce: bool = True,
        bucket_size_mb: int = -1,
        loss_scale: Union[float, str, dict] = 512.0,
        distributed: bool = True,
    ) -> None:
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == "dynamic":
            self.loss_scaler = XPUGradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = XPUGradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = XPUGradScaler(**loss_scale)
        else:
            raise ValueError("loss_scale must be of type float, dict, or " f'"dynamic", got {loss_scale}')
