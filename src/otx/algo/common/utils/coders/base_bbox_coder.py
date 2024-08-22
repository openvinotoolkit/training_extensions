# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base bounding box coder."""

from abc import abstractmethod

from torch import Tensor


class BaseBBoxCoder:
    """Base class for bounding box coder."""

    @abstractmethod
    def encode(self, *args, **kwargs) -> Tensor:
        """Encode bounding boxes."""

    @abstractmethod
    def decode(self, *args, **kwargs) -> Tensor:
        """Decode bounding boxes."""

    @abstractmethod
    def decode_export(self, *args, **kwargs) -> Tensor:
        """Decode bounding boxes for export."""
