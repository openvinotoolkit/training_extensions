"""Custom GradScaler to scale loss."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import abc, defaultdict
from typing import List

import torch

from otx.algorithms.common.utils.utils import is_xpu_available

if is_xpu_available():
    from intel_extension_for_pytorch.cpu.autocast._grad_scaler import _MultiDeviceReplicator
from torch.cuda.amp.grad_scaler import GradScaler, _refresh_per_optimizer_state


class XPUGradScaler(GradScaler):
    """GradScaler for XPU."""

    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._enabled = enabled
        if not is_xpu_available():
            raise RuntimeError("XPU GradScaler requires XPU device.")

        if self._enabled:
            assert growth_factor > 1.0, "The growth factor must be > 1.0."
            assert backoff_factor < 1.0, "The backoff factor must be < 1.0."

            self._init_scale = init_scale
            # self._scale will be lazily initialized during the first call to scale()
            self._scale = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._init_growth_tracker = 0
            # self._growth_tracker will be lazily initialized during the first call to scale()
            self._growth_tracker = None
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def scale(self, outputs):
        """Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            assert outputs.device.type == "xpu"
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: List[_MultiDeviceReplicator] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.device.type == "xpu"
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(_MultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, abc.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_bf16=False):
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        per_device_found_inf = _MultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be hundreds of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled bf16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.bfloat16:
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    # TODO: is there a way to split by device and dtype without appending in the inner loop?
                    per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads, per_device_found_inf.get(device), per_device_inv_scale.get(device)
                    )

        return per_device_found_inf._per_device_tensors
