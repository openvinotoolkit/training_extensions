"""This module implements the Coordinate entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import Tuple


class Coordinate:
    """Represents a 2D-coordinate with an x-position and a y-position.

    NB most coordinates are normalized (between 0.0 and 1.0)

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
    """

    __slots__ = ["x", "y"]

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        """String representation of the coordinate. Returns the x and y coordinates."""
        return f"Coordinate(x={self.x}, y={self.y})"

    def __eq__(self, other):
        """Returns True if the coordinates are equal."""
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        """Returns the hash of the coordinate."""
        return hash(str(self))

    def as_tuple(self) -> Tuple[float, float]:
        """Convert the coordinates to a pair (x,y)."""
        return self.x, self.y

    def as_int_tuple(self) -> Tuple[int, int]:
        """Convert the coordinates to a pair of integer coordinates (x,y)."""
        return int(self.x), int(self.y)
