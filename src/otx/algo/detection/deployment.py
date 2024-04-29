"""Deployment functions for detection algorithms."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import torch

# ruff: noqa: A002


def gather_topk_(
    *inputs: tuple[torch.Tensor],
    inds: torch.Tensor,
    batch_size: int,
    is_batched: bool = True,
) -> list[torch.Tensor] | torch.Tensor:
    """Gather topk of each tensor.

    Args:
        inputs (tuple[torch.Tensor]): Tensors to be gathered.
        inds (torch.Tensor): Topk index.
        batch_size (int): batch_size.
        is_batched (bool): Inputs is batched or not.

    Returns:
        Tuple[torch.Tensor]: Gathered tensors.
    """
    outputs = __gather_topk(
        *inputs,
        inds=inds,
        batch_size=batch_size,
        is_batched=is_batched,
    )

    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def __gather_topk(
    *inputs: tuple[torch.Tensor],
    inds: torch.Tensor,
    batch_size: int,
    is_batched: bool = True,
) -> list[torch.Tensor]:
    """The default implementation of gather_topk."""
    if is_batched:
        batch_inds = torch.arange(batch_size, device=inds.device).unsqueeze(-1)
        outputs = [x[batch_inds, inds, ...] if x is not None else None for x in inputs]  # type: ignore[call-overload]
    else:
        prior_inds = inds.new_zeros((1, 1))
        outputs = [x[prior_inds, inds, ...] if x is not None else None for x in inputs]  # type: ignore[call-overload]

    return outputs
