"""This module implements the Tensor entity."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Tuple

import numpy as np

from otx.api.entities.metadata import IMetadata


class TensorEntity(IMetadata):
    """Represents a metadata of tensor type in OTX.

    Args:
        name: name of metadata
        numpy: the numpy data of the tensor
    """

    def __init__(self, name: str, numpy: np.ndarray):
        self.name = name
        # Copying Numpy array as it points to the same memory address
        self._numpy = np.copy(numpy)

    @property
    def numpy(self) -> np.ndarray:
        """Returns the numpy representation of the tensor."""
        return self._numpy

    @numpy.setter
    def numpy(self, value):
        self._numpy = value

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the tensor."""
        return self._numpy.shape

    def __eq__(self, other):
        """Returns True if the tensors are equal."""
        if isinstance(other, TensorEntity):
            return np.array_equal(self.numpy, other.numpy)
        return False

    def __str__(self):
        """Returns the string representation of the tensor."""
        return f"{self.__class__.__name__}(name={self.name}, shape={self.shape})"

    def __repr__(self):
        """Returns the representation of the tensor."""
        return f"{self.__class__.__name__}(name={self.name})"
