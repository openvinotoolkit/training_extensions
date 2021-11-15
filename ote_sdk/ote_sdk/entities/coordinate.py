"""This module implements the Coordinate entity"""
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


class Coordinate:
    """
    Represents a 2D-coordinate with an x-position and a y-position.

    NB most coordinates are normalized (between 0.0 and 1.0)

    :param x: x-coordinate
    :param y: y-coordinate
    """

    __slots__ = ["x", "y"]

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Coordinate(x={self.x}, y={self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(str(self))

    def as_tuple(self) -> Tuple[float, float]:
        """
        Convert the coordinates to a pair (x,y)
        """
        return self.x, self.y

    def as_int_tuple(self) -> Tuple[int, int]:
        """
        Convert the coordinates to a pair of integer coordinates (x,y)
        """
        return int(self.x), int(self.y)
