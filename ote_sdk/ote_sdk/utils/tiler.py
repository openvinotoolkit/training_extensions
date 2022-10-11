"""
Tiling Module
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from itertools import product

import numpy as np


class Tiler:
    """Tile Image into (non)overlapping Patches. Images are tiled in order to efficiently process large images.

    Args:
        tile_size: Tile dimension for each patch
        overlap: Overlap between adjacent tile
        batch_size: Batch Size
    """

    def __init__(self, tile_size: int, overlap: float, batch_size=1) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.batch_size = batch_size

    def tile(self, image: np.ndarray) -> np.ndarray:
        """Tiles an input image to either overlapping, non-overlapping or random patches.

        Args:
            image: Input image to tile.

        Returns:
            Tiles generated from the image.
        """
        height, width = image.shape[:2]

        tiles, offsets = [], []
        for (loc_i, loc_j) in product(
            range(0, height - self.tile_size + 1, self.stride),
            range(0, width - self.tile_size + 1, self.stride),
        ):
            tiles.append(
                image[
                    loc_i : (loc_i + self.tile_size),
                    loc_j : (loc_j + self.tile_size),
                    ...,
                ]
            )
            offsets.append((loc_i, loc_j))  # offset y, offset x
            if len(tiles) == self.batch_size:
                yield tiles, offsets
                tiles, offsets = [], []
        if tiles:
            yield tiles, offsets
