"""
Tiling Module
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List
from itertools import product

import numpy as np


class Tiler:
    """Tile Image into (non)overlapping Patches. Images are tiled in order to efficiently process large images.

    Args:
        tile_size: Tile dimension for each patch
        overlap: Overlap between adjacent tile
    """

    def __init__(self, tile_size: int, overlap: float) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))

    def tile(self, image: np.ndarray) -> List[List]:
        """Tiles an input image to either overlapping, non-overlapping or random patches.

        Args:
            image: Input image to tile.

        Returns:
            Tiles coordinates
        """
        height, width = image.shape[:2]

        coords = [(0, 0, width, height)]
        for (loc_j, loc_i) in product(
            range(0, width - self.tile_size + 1, self.stride),
            range(0, height - self.tile_size + 1, self.stride),
        ):
            coords.append((loc_j, loc_i, loc_j + self.tile_size, loc_i + self.tile_size))
        return coords
