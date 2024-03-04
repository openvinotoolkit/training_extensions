# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from datumaro import Dataset as DmDataset
from datumaro import Image
from datumaro.plugins.tiling.util import xywh_to_x1y1x2y2
from omegaconf import DictConfig, OmegaConf
from openvino.model_api.models import Model
from openvino.model_api.tilers import DetectionTiler, InstanceSegmentationTiler, Tiler
from otx.core.config.data import (
    DataModuleConfig,
    SubsetConfig,
    TileConfig,
    VisualPromptingConfig,
)
from otx.core.data.dataset.tile import OTXTileTransform
from otx.core.data.entity.detection import DetBatchDataEntity
from otx.core.data.entity.tile import TileBatchDetDataEntity
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.detection import OVDetectionModel
from otx.core.model.entity.instance_segmentation import OVInstanceSegmentationModel
from otx.core.types.task import OTXTaskType
from otx.engine import Engine


def test_tile_transform_consistency(mocker):
    # Test that the tiler and tile transform are consistent
    rng = np.random.default_rng()
    rnd_tile_size = rng.integers(low=100, high=500)
    rnd_tile_overlap = rng.random()
    image_size = rng.integers(low=1000, high=5000)
    np_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    dm_image = Image.from_numpy(np_image)

    mock_model = MagicMock(spec=Model)
    mocker.patch("openvino.model_api.tilers.tiler.Tiler.__init__", return_value=None)
    mocker.patch.multiple(Tiler, __abstractmethods__=set())

    tiler = Tiler(model=mock_model)
    tiler.tile_size = rnd_tile_size
    tiler.tiles_overlap = rnd_tile_overlap

    mocker.patch("otx.core.data.dataset.tile.OTXTileTransform.__init__", return_value=None)
    tile_transform = OTXTileTransform()
    tile_transform._tile_size = (rnd_tile_size, rnd_tile_size)
    tile_transform._overlap = (rnd_tile_overlap, rnd_tile_overlap)

    dm_rois = [xywh_to_x1y1x2y2(*roi) for roi in tile_transform._extract_rois(dm_image)]
    # 0 index in tiler is the full image so we skip it
    assert np.allclose(dm_rois, tiler._tile(np_image)[1:])