# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OVInstanceSegmentationModel tiled inference (forward_tiles delegation)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from getitune.backend.openvino.models.instance_segmentation import OVInstanceSegmentationModel
from getitune.config.data import TileConfig
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import PredictionBatch
from getitune.data.entity.tile import TileBatchData


class _FakeInstanceSegResult:
    """Mimics model_api InstanceSegmentationResult with numpy arrays."""

    def __init__(self, bboxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, masks: np.ndarray):
        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels
        self.masks = masks
        self.saliency_map = []
        self.feature_vector = np.ndarray(0)


class _FakeTileInfo:
    """Minimal datumaro-TileInfo stand-in exposing tile placement fields."""

    def __init__(self, x: int, y: int, width: int, height: int, source_sample_idx: int = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.source_sample_idx = source_sample_idx


class _FakeTileBatchInstSeg(TileBatchData):
    """Minimal TileBatchData subclass exposing the fields forward_tiles reads."""

    def __init__(self, imgs_info, batch_tiles, batch_tile_tile_infos):
        self.imgs_info = imgs_info
        self.batch_tiles = batch_tiles
        self.batch_tile_tile_infos = batch_tile_tile_infos


class TestOVInstanceSegmentationModelTiling:
    """Tests for tile-aware inference in OVInstanceSegmentationModel (forward_tiles delegation)."""

    @pytest.fixture
    def inst_seg_model(self):
        with patch.object(OVInstanceSegmentationModel, "__init__", lambda *_args, **_kwargs: None):
            model = OVInstanceSegmentationModel.__new__(OVInstanceSegmentationModel)
            model.model = MagicMock()
            model.tile_config = TileConfig(enable_tiler=True, tile_size=(200, 200))
            model._label_info = MagicMock()
            model._label_info.num_classes = 3
            model.hparams = {}
            return model

    def _make_tile_batch(self) -> _FakeTileBatchInstSeg:
        ori_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(256, 256),
            ori_shape=(256, 256),
        )
        batch_tiles = [[torch.rand(3, 200, 200), torch.rand(3, 200, 200)]]
        tile_infos = [
            _FakeTileInfo(x=0, y=0, width=200, height=200),
            _FakeTileInfo(x=56, y=56, width=200, height=200),
        ]
        return _FakeTileBatchInstSeg(
            imgs_info=[ori_info],
            batch_tiles=batch_tiles,
            batch_tile_tile_infos=[tile_infos],
        )

    def test_forward_tiles_delegates_to_model_api_tiler(self, inst_seg_model):
        """forward_tiles feeds native tiles + coords to the ModelAPI tiler and converts the merged result."""
        tile_batch = self._make_tile_batch()

        merged = _FakeInstanceSegResult(
            bboxes=np.array([[10, 10, 50, 50]], dtype=np.int32),
            scores=np.array([0.9], dtype=np.float32),
            labels=np.array([2], dtype=np.int32),  # 1-based ModelAPI label -> getitune label 1
            masks=np.ones((1, 256, 256), dtype=np.uint8),
        )
        tiler = MagicMock()
        tiler.predict_tiles.return_value = merged
        inst_seg_model._get_tiler = MagicMock(return_value=tiler)

        result = inst_seg_model.forward_tiles(tile_batch)

        tiler.predict_tiles.assert_called_once()
        call_tiles, call_coords, call_shape = tiler.predict_tiles.call_args.args
        assert len(call_tiles) == 2
        assert call_tiles[0].shape == (200, 200, 3)
        assert call_coords == [[0, 0, 200, 200], [56, 56, 256, 256]]
        assert call_shape == (256, 256, 3)

        assert isinstance(result, PredictionBatch)
        torch.testing.assert_close(
            result.bboxes[0].data,  # pyrefly: ignore[unsupported-operation]
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        )
        torch.testing.assert_close(result.scores[0], torch.tensor([0.9]))  # pyrefly: ignore[unsupported-operation]
        # ModelAPI 1-based label shifted down to getitune 0-based.
        torch.testing.assert_close(result.labels[0], torch.tensor([1]))  # pyrefly: ignore[unsupported-operation]
        assert result.masks is not None
        assert result.masks[0].shape == (1, 256, 256)  # pyrefly: ignore[unsupported-operation]

    def test_forward_tiles_empty_masks_yield_none(self, inst_seg_model):
        """When no instances survive merging, masks must be reported as None."""
        tile_batch = self._make_tile_batch()
        merged = _FakeInstanceSegResult(
            bboxes=np.empty((0, 4), dtype=np.int32),
            scores=np.empty((0,), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int32),
            masks=np.empty((0, 256, 256), dtype=np.uint8),
        )
        tiler = MagicMock()
        tiler.predict_tiles.return_value = merged
        inst_seg_model._get_tiler = MagicMock(return_value=tiler)

        result = inst_seg_model.forward_tiles(tile_batch)
        assert result.masks is None
        assert result.bboxes[0].shape == (0, 4)  # pyrefly: ignore[unsupported-operation]

    def test_get_tiler_reuses_existing_model_api_tiler(self, inst_seg_model):
        """_get_tiler returns the already-wrapped ModelAPI tiler without constructing a new one."""
        from model_api.tilers import InstanceSegmentationTiler

        existing = InstanceSegmentationTiler.__new__(InstanceSegmentationTiler)
        inst_seg_model.model = existing
        assert inst_seg_model._get_tiler() is existing
