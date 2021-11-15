"""This module implements the Tensor entity"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

from typing import Tuple

import numpy as np

from ote_sdk.entities.metadata import IMetadata


class TensorEntity(IMetadata):
    """
    Represents a metadata of tensor type in OTE.

    :param name: name of metadata
    :param numpy: the numpy data of the tensor
    """

    def __init__(self, name: str, numpy: np.ndarray):
        self.name = name
        self._numpy = numpy

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
        if isinstance(other, TensorEntity):
            return np.array_equal(self.numpy, other.numpy)
        return False

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, shape={self.shape})"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, numpy={self.numpy})"
