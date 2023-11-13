"""PSP module."""

# Copyright (c) 2019 MendelXu
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn


class PSPModule(nn.Module):
    """Pyramid Scene Parsing module (PSP module) implementation. Reference: https://github.com/MendelXu/ANN."""

    methods: ClassVar = {"max": nn.AdaptiveMaxPool2d, "avg": nn.AdaptiveAvgPool2d}

    def __init__(self, sizes: tuple[int, ...] = (1, 3, 6, 8), method: str = "max") -> None:
        """Initializes PSPModule.

        Args:
            sizes (tuple[int, ...]): Output sizes of the pooling layers.
            method (str): Pooling method to use. Must be one of "max", "avg", or "adaptive_avg".

        Raises:
            NotImplementedError: If an invalid pooling method is specified.

        Attributes:
            methods (dict): Mapping of supported pooling methods to their corresponding pooling block classes.
            stages (nn.ModuleList): List of pooling blocks in the PSP module.
        """
        super().__init__()

        if method not in self.methods:
            msg = f"Method must be one of {self.methods.keys()}."
            raise NotImplementedError(msg)

        pool_block = self.methods[method]

        self.stages = nn.ModuleList([pool_block(output_size=(size, size)) for size in sizes])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward."""
        batch_size, c, _, _ = feats.size()

        priors = [stage(feats).view(batch_size, c, -1) for stage in self.stages]

        return torch.cat(priors, -1)
