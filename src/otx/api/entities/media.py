"""This module implements the Media entity."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Optional

import numpy as np

from otx.api.entities.annotation import Annotation


class IMediaEntity(metaclass=abc.ABCMeta):
    """Media entity interface.

    This interface is used to represent any kind of media data, on which users can annotate and tasks can perform
    training/analysis.
    """


class IMedia2DEntity(IMediaEntity, metaclass=abc.ABCMeta):
    """This interface is used to represent IMedia which is 2-dimensional media, i.e., containing height and width."""

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
        """Returns the numpy representation of the 2D Media object while taking the roi into account."""
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

    @property
    def path(self) -> Optional[str]:
        """Returns the path of the 2D Media object."""
        return None
