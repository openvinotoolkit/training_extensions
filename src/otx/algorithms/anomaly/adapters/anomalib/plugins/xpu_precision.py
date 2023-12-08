"""Plugin for mixed-precision training on XPU"""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, Union

import torch
from torch import Tensor
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from lightning_fabric.utilities.types import Optimizable
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MixedPrecisionXPUPlugin(PrecisionPlugin):
    """Plugin for Automatic Mixed Precision (AMP) training with ``torch.xpu.autocast``.

    Args:
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    def __init__(
        self, scaler: Optional[Any] = None
    ) -> None:
        self.scaler = scaler

    def pre_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        return super().pre_backward(tensor, module)

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(
                optimizer, model=model, optimizer_idx=optimizer_idx, closure=closure, **kwargs
            )
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()

        if not _optimizer_handles_unscaling(optimizer):
            # Unscaling needs to be performed here in case we are going to apply gradient clipping.
            # Optimizers that perform unscaling in their `.step()` method are not supported (e.g., fused Adam).
            # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
            self.scaler.unscale_(optimizer)

        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)
            self.scaler.update()
            return step_output
        return closure_result

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        if clip_val > 0 and _optimizer_handles_unscaling(optimizer):
            raise RuntimeError(
                f"The current optimizer, {type(optimizer).__qualname__}, does not allow for gradient clipping"
                " because it performs unscaling of gradients internally. HINT: Are you using a 'fused' optimizer?"
            )
        super().clip_gradients(optimizer=optimizer, clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with torch.xpu.autocast(True):
            yield

    def state_dict(self) -> Dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)


def _optimizer_handles_unscaling(optimizer: Any) -> bool:
    """Determines whether a PyTorch optimizer handles unscaling gradients in the step method rather than through the
    :class:`torch.cuda.amp.GradScaler`.

    Since, the current implementation of this function checks a PyTorch internal variable on the optimizer, the return
    value will only be reliable for built-in PyTorch optimizers.
    """
    return getattr(optimizer, "_step_supports_amp_scaling", False)
