"""This module implements the Media entity"""

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

import abc
from typing import Optional

import numpy as np

from ote_sdk.entities.annotation import Annotation


class IMediaEntity(metaclass=abc.ABCMeta):
    """
    This interface is used to represent any kind of media data, on which users can
    annotate and tasks can perform training/analysis.
    """


class IMedia2DEntity(IMediaEntity, metaclass=abc.ABCMeta):
    """
    This interface is used to represent IMedia which is 2-dimensional media,
    i.e., containing height and width.
    """

    @property  # type:ignore
    @abc.abstractmethod
    def numpy(self) -> np.ndarray:
        """Returns the numpy representation of the 2D Media object."""
        raise NotImplementedError

    @numpy.setter  # type:ignore
    @abc.abstractmethod
    def numpy(self, value: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def roi_numpy(self, roi: Optional[Annotation]) -> np.ndarray:
        """
        Returns the numpy representation of the 2D Media object while taking the roi
        into account.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """Returns the height of the 2D Media object."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """Returns the width representation of the 2D Media object."""
        raise NotImplementedError
