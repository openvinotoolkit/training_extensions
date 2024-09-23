# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from datumaro import Image
from datumaro.plugins.tiling.util import xywh_to_x1y1x2y2
from model_api.models import Model
from model_api.tilers import Tiler
from otx.core.data.dataset.tile import OTXTileTransform


def test_tile_transform_consistency(mocker):
    # Test that OV tiler and PyTorch tile transform are consistent
    rng = np.random.default_rng()
    rnd_tile_size = rng.integers(low=100, high=500)
    rnd_tile_overlap = rng.random()
    image_size = rng.integers(low=1000, high=5000)
    np_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    dm_image = Image.from_numpy(np_image)

    mock_model = MagicMock(spec=Model)
    mocker.patch("model_api.tilers.tiler.Tiler.__init__", return_value=None)
    mocker.patch.multiple(Tiler, __abstractmethods__=set())

    tiler = Tiler(model=mock_model)
    tiler.tile_with_full_img = True
    tiler.tile_size = rnd_tile_size
    tiler.tiles_overlap = rnd_tile_overlap

    mocker.patch("otx.core.data.dataset.tile.OTXTileTransform.__init__", return_value=None)
    tile_transform = OTXTileTransform()
    tile_transform._tile_size = (rnd_tile_size, rnd_tile_size)
    tile_transform._overlap = (rnd_tile_overlap, rnd_tile_overlap)
    tile_transform.with_full_img = True

    dm_rois = [xywh_to_x1y1x2y2(*roi) for roi in tile_transform._extract_rois(dm_image)]
    ov_tiler_rois = tiler._tile(np_image)

    assert len(dm_rois) == len(ov_tiler_rois)
    for dm_roi in dm_rois:
        assert list(dm_roi) in ov_tiler_rois
