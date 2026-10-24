# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base bounding box coder."""

from abc import ABCMeta, abstractmethod

from torch import Tensor


class BaseBBoxCoder(metaclass=ABCMeta):
    """Base class for bounding box coder."""

    encode_size: int

    @abstractmethod
    def encode(self, *args, **kwargs) -> Tensor:
        """Encode bounding boxes."""

    @abstractmethod
    def decode(self, *args, **kwargs) -> Tensor:
        """Decode bounding boxes."""

    @abstractmethod
    def decode_export(self, *args, **kwargs) -> Tensor:
        """Decode bounding boxes for export."""
