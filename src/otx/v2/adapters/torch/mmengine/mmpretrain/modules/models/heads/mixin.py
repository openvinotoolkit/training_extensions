"""Module defining Mix-in class of heads."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Union

import torch


class OTXHeadMixin:
    """Mix-in class for OTX custom heads."""

    @staticmethod
    def pre_logits(x: Union[list, torch.Tensor]) -> torch.Tensor:
        """Preprocess logits before forward. Designed to support vision transformer output."""
        if isinstance(x, list):
            x = x[-1]
            return x
        return x
