# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for OTX algo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from torch import nn


def _torch_hub_model_reduce(self) -> tuple[Callable, tuple]:  # noqa: ANN001
    return (torch_hub_load, self.torch_hub_load_args)


def torch_hub_load(repo_or_dir: str, model: str) -> nn.Module:
    """Load a module using from 'torch.hub'. The module is modified to support pickle."""
    module = torch.hub.load(
        repo_or_dir=repo_or_dir,
        model=model,
    )

    # support pickle
    module.torch_hub_load_args = (repo_or_dir, model)
    module.__class__.__reduce__ = _torch_hub_model_reduce.__get__(module, module.__class__)
    return module
