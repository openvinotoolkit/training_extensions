"""Image Tiler."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from itertools import product
from typing import Sequence, Union

import numpy as np


class Tiler:
    def __init__(self, tile_size: int, overlap: float, batch_size=1) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.batch_size = batch_size

    def tile(self, image: np.ndarray) -> np.ndarray:
        """Tiles an input image to either overlapping, non-overlapping or random patches.

        Args:
            image: Input image to tile.

        Examples:
            >>> from anomalib.data.tiler import Tiler
            >>> tiler = Tiler(tile_size=512,stride=256)
            >>> image = torch.rand(size=(2, 3, 1024, 1024))
            >>> image.shape
            torch.Size([1024, 1024, 3])
            >>> tiles = tiler.tile(image)
            >>> tiles.shape
            torch.Size([18, 512, 512, 3])

        Returns:
            Tiles generated from the image.
        """
        height, width, channels = image.shape

        self.num_patches_h = int((height - self.tile_size) / self.stride) + 1
        self.num_patches_w = int((width - self.tile_size) / self.stride) + 1

        tiles, offsets = [], []
        for (tile_i, tile_j), (loc_i, loc_j) in zip(
            product(range(self.num_patches_h), range(self.num_patches_w)),
            product(
                range(0, height - self.tile_size + 1, self.stride),
                range(0, width - self.tile_size + 1, self.stride),
            ),
        ):
            tiles.append(image[loc_i : (loc_i + self.tile_size), loc_j : (loc_j + self.tile_size), ...])
            offsets.append((loc_i, loc_j))  # offset y, offset x
            if len(tiles) == self.batch_size:
                yield tiles, offsets
                tiles, offsets = [], []
        if len(tiles):
            yield tiles, offsets
