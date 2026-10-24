# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OVDetectionModel._customize_outputs coordinate rescaling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchvision import tv_tensors

from getitune.backend.openvino.models.detection import OVDetectionModel
from getitune.config.data import TileConfig
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.data.entity.tile import TileBatchData


class _FakeDetectionResult:
    """Mimics model_api DetectionResult with numpy arrays."""

    def __init__(self, bboxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels
        self.saliency_map = np.ndarray(0)
        self.feature_vector = np.ndarray(0)


class TestOVDetectionModelCoordinateRescaling:
    """Tests that _customize_outputs correctly rescales bboxes from img_shape to ori_shape."""

    @pytest.fixture
    def detection_model(self):
        """Create a minimally mocked OVDetectionModel (no real IR needed)."""
        with patch.object(OVDetectionModel, "__init__", lambda *_args, **_kwargs: None):
            model = OVDetectionModel.__new__(OVDetectionModel)
            # Mock the model attribute that _customize_outputs uses for label names
            mock_model = MagicMock()
            mock_model.get_label_name.return_value = "class_0"
            model.model = mock_model
            return model

    def _make_sample_batch(
        self,
        batch_size: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        scale_factor: tuple[float, float] | None = (1.0, 1.0),
    ) -> SampleBatch:
        """Create a SampleBatch with specified img_shape and ori_shape."""
        images = [torch.rand(3, *img_shape) for _ in range(batch_size)]
        imgs_info = [
            ImageInfo(
                img_idx=i,
                img_shape=img_shape,
                ori_shape=ori_shape,
                padding=padding,
                scale_factor=scale_factor,
            )
            for i in range(batch_size)
        ]
        return SampleBatch(images=images, imgs_info=imgs_info)

    def test_rescales_bboxes_when_shapes_differ(self, detection_model):
        """When img_shape != ori_shape, bboxes should be rescaled to ori_shape."""
        img_shape = (640, 640)
        ori_shape = (1024, 768)  # (H, W) - original image is 768x1024

        # Simulated ModelAPI output: bboxes in img_shape (640x640) pixel coords
        bboxes_in_img_space = np.array(
            [[100.0, 200.0, 300.0, 400.0], [50.0, 50.0, 600.0, 600.0]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int32)

        outputs = [_FakeDetectionResult(bboxes_in_img_space, scores, labels)]
        inputs = self._make_sample_batch(batch_size=1, img_shape=img_shape, ori_shape=ori_shape)

        result = detection_model._customize_outputs(outputs, inputs)

        assert isinstance(result, PredictionBatch)
        result_bboxes = result.bboxes[0]

        # Expected rescaling factors
        scale_x = ori_shape[1] / img_shape[1]  # 768 / 640 = 1.2
        scale_y = ori_shape[0] / img_shape[0]  # 1024 / 640 = 1.6

        expected_bboxes = torch.tensor(
            [
                [100.0 * scale_x, 200.0 * scale_y, 300.0 * scale_x, 400.0 * scale_y],
                [50.0 * scale_x, 50.0 * scale_y, 600.0 * scale_x, 600.0 * scale_y],
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(result_bboxes.data, expected_bboxes)
        assert result_bboxes.canvas_size == ori_shape

    def test_scores_are_float32_when_modelapi_returns_float64(self, detection_model):
        """Scores must be float32 to match float32 bboxes.

        ModelAPI returns numpy arrays whose default float dtype is float64. If scores were
        left as float64, torchvision NMS during tile merge fails with
        "dets should have the same type as scores". This regression test locks the dtype.
        """
        img_shape = (640, 640)
        ori_shape = (640, 640)

        bboxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        # ModelAPI default float dtype is float64.
        scores = np.array([0.9], dtype=np.float64)
        labels = np.array([0], dtype=np.int64)

        outputs = [_FakeDetectionResult(bboxes, scores, labels)]
        inputs = self._make_sample_batch(batch_size=1, img_shape=img_shape, ori_shape=ori_shape)

        result = detection_model._customize_outputs(outputs, inputs)

        assert result.scores is not None
        assert result.bboxes is not None
        assert result.scores[0].dtype == torch.float32
        # Scores and bboxes share a dtype so torchvision NMS won't raise during tile merge.
        assert result.scores[0].dtype == result.bboxes[0].dtype

    def test_no_rescaling_when_shapes_equal(self, detection_model):
        """When img_shape == ori_shape, bboxes should remain unchanged."""
        shape = (640, 640)

        bboxes = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
        scores = np.array([0.95], dtype=np.float32)
        labels = np.array([0], dtype=np.int32)

        outputs = [_FakeDetectionResult(bboxes, scores, labels)]
        inputs = self._make_sample_batch(batch_size=1, img_shape=shape, ori_shape=shape)

        result = detection_model._customize_outputs(outputs, inputs)
        result_bboxes = result.bboxes[0]

        expected = torch.tensor([[100.0, 200.0, 300.0, 400.0]], dtype=torch.float32)
        torch.testing.assert_close(result_bboxes.data, expected)
        assert result_bboxes.canvas_size == shape

    def test_empty_detections(self, detection_model):
        """Empty detections should not cause errors."""
        img_shape = (640, 640)
        ori_shape = (1080, 1920)

        bboxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)

        outputs = [_FakeDetectionResult(bboxes, scores, labels)]
        inputs = self._make_sample_batch(batch_size=1, img_shape=img_shape, ori_shape=ori_shape)

        result = detection_model._customize_outputs(outputs, inputs)
        assert result.bboxes[0].shape == (0, 4)
        assert result.bboxes[0].canvas_size == ori_shape

    def test_batch_with_different_original_sizes(self, detection_model):
        """Each image in the batch may have a different ori_shape."""
        img_shape = (640, 640)
        ori_shapes = [(800, 600), (1200, 1600)]

        images = [torch.rand(3, *img_shape) for _ in range(2)]
        imgs_info = [ImageInfo(img_idx=i, img_shape=img_shape, ori_shape=ori_shapes[i]) for i in range(2)]
        inputs = SampleBatch(images=images, imgs_info=imgs_info)

        outputs = [
            _FakeDetectionResult(
                np.array([[320.0, 320.0, 640.0, 640.0]], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
                np.array([0], dtype=np.int32),
            ),
            _FakeDetectionResult(
                np.array([[0.0, 0.0, 640.0, 640.0]], dtype=np.float32),
                np.array([0.85], dtype=np.float32),
                np.array([1], dtype=np.int32),
            ),
        ]

        result = detection_model._customize_outputs(outputs, inputs)

        # Image 0: ori (800, 600) → scale_x = 600/640, scale_y = 800/640
        scale_x_0 = 600 / 640
        scale_y_0 = 800 / 640
        expected_0 = torch.tensor(
            [[320.0 * scale_x_0, 320.0 * scale_y_0, 640.0 * scale_x_0, 640.0 * scale_y_0]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result.bboxes[0].data, expected_0)
        assert result.bboxes[0].canvas_size == (800, 600)

        # Image 1: ori (1200, 1600) → scale_x = 1600/640, scale_y = 1200/640
        scale_x_1 = 1600 / 640
        scale_y_1 = 1200 / 640
        expected_1 = torch.tensor(
            [[0.0, 0.0, 640.0 * scale_x_1, 640.0 * scale_y_1]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result.bboxes[1].data, expected_1)
        assert result.bboxes[1].canvas_size == (1200, 1600)

    def test_metric_alignment_preds_vs_targets(self, detection_model):
        """Verify predictions and targets end up in the same coordinate space for metric computation.

        This is the core bug scenario: OV transforms resize images to img_shape,
        but ground truth targets remain in ori_shape (resize_targets=false).
        After the fix, predicted bboxes should be in ori_shape coords too.
        """
        img_shape = (640, 640)
        ori_shape = (480, 854)  # typical 480p image (H=480, W=854)

        # Ground truth in ori_shape coords (as provided by dataset with resize_targets=false)
        gt_bboxes = tv_tensors.BoundingBoxes(
            data=torch.tensor([[100.0, 50.0, 400.0, 300.0]]),
            format="XYXY",
            canvas_size=ori_shape,
        )
        gt_labels = torch.tensor([0], dtype=torch.long)

        # ModelAPI produces bboxes in img_shape coords (the bug: it sees 640x640 as "original")
        # A perfect detection of the GT box would look like this in img_shape coords:
        scale_x_inv = img_shape[1] / ori_shape[1]  # 640 / 854
        scale_y_inv = img_shape[0] / ori_shape[0]  # 640 / 480
        pred_bboxes_in_img = np.array(
            [[100.0 * scale_x_inv, 50.0 * scale_y_inv, 400.0 * scale_x_inv, 300.0 * scale_y_inv]],
            dtype=np.float32,
        )
        pred_scores = np.array([0.99], dtype=np.float32)
        pred_labels = np.array([0], dtype=np.int32)

        outputs = [_FakeDetectionResult(pred_bboxes_in_img, pred_scores, pred_labels)]
        inputs = self._make_sample_batch(batch_size=1, img_shape=img_shape, ori_shape=ori_shape)
        # Attach ground truth to inputs for metric preparation
        inputs.bboxes = [gt_bboxes]
        inputs.labels = [gt_labels]

        result = detection_model._customize_outputs(outputs, inputs)

        # After fix: predicted bboxes should be back in ori_shape coords
        # matching the ground truth
        torch.testing.assert_close(
            result.bboxes[0].data,
            gt_bboxes.data,
            atol=1e-4,
            rtol=1e-4,
        )

        # Verify metric inputs are aligned
        metric_inputs = detection_model.prepare_metric_inputs(result, inputs)
        pred_boxes = metric_inputs["preds"][0]["boxes"]
        target_boxes = metric_inputs["target"][0]["boxes"]
        torch.testing.assert_close(pred_boxes, target_boxes, atol=1e-4, rtol=1e-4)

    def test_letterbox_rescaling(self, detection_model):
        """When preprocessing uses letterbox (padding + aspect-ratio resize), bboxes should be unpadded and unscaled."""
        # Original image: 480x640 (H, W)
        # Letterbox to 640x640: scale=1.0 on height (480*1.0=480 padded to 640), scale=1.0 on width (640)
        # Actually: fit_to_window scales uniformly by min(640/480, 640/640) = 1.0
        # so image stays 480x640, padded to 640x640 with padding (0, 80, 0, 80) top/bottom
        # More realistic: original 1080x1920, target 640x640
        # scale = min(640/1080, 640/1920) = 0.333
        # scaled = 360x640, padding_left=0, padding_top=140
        ori_shape = (1080, 1920)
        img_shape = (640, 640)
        scale = min(img_shape[0] / ori_shape[0], img_shape[1] / ori_shape[1])  # 0.333
        scaled_h = int(ori_shape[0] * scale)  # 360
        pad_top = (img_shape[0] - scaled_h) // 2  # 140
        padding = (0, pad_top, 0, img_shape[0] - scaled_h - pad_top)  # (left, top, right, bottom)
        scale_factor = (scale, scale)  # (h_scale, w_scale)

        # Bbox in img_shape coords (after letterbox): x=100, y=200 (includes padding)
        bboxes_in_img = np.array([[100.0, 200.0, 500.0, 400.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        labels = np.array([0], dtype=np.int32)

        outputs = [_FakeDetectionResult(bboxes_in_img, scores, labels)]
        inputs = self._make_sample_batch(
            batch_size=1,
            img_shape=img_shape,
            ori_shape=ori_shape,
            padding=padding,
            scale_factor=scale_factor,
        )

        result = detection_model._customize_outputs(outputs, inputs)
        result_bboxes = result.bboxes[0]

        # Expected: undo padding then divide by scale
        expected_x1 = (100.0 - padding[0]) / scale
        expected_y1 = (200.0 - padding[1]) / scale
        expected_x2 = (500.0 - padding[0]) / scale
        expected_y2 = (400.0 - padding[1]) / scale
        # Clamp to ori_shape
        expected_x1 = max(0.0, min(expected_x1, float(ori_shape[1])))
        expected_y1 = max(0.0, min(expected_y1, float(ori_shape[0])))
        expected_x2 = max(0.0, min(expected_x2, float(ori_shape[1])))
        expected_y2 = max(0.0, min(expected_y2, float(ori_shape[0])))

        expected = torch.tensor([[expected_x1, expected_y1, expected_x2, expected_y2]], dtype=torch.float32)
        torch.testing.assert_close(result_bboxes.data, expected, atol=1e-4, rtol=1e-4)
        assert result_bboxes.canvas_size == ori_shape


class _FakeTileInfo:
    """Minimal datumaro-TileInfo stand-in exposing tile placement fields."""

    def __init__(self, x: int, y: int, width: int, height: int, source_sample_idx: int = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.source_sample_idx = source_sample_idx


class _FakeTileBatchDet(TileBatchData):
    """Minimal TileBatchData subclass exposing the fields forward_tiles reads."""

    def __init__(self, imgs_info, batch_tiles, batch_tile_tile_infos):
        # Bypass the dataclass __init__: forward_tiles only reads these attributes.
        self.imgs_info = imgs_info
        self.batch_tiles = batch_tiles
        self.batch_tile_tile_infos = batch_tile_tile_infos
        # Ground-truth fields consumed by prepare_metric_inputs during test_step.
        self.bboxes = [
            tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                torch.empty((0, 4)),
                format="XYXY",
                canvas_size=info.ori_shape,
            )
            for info in imgs_info
        ]
        self.labels = [torch.empty((0,), dtype=torch.long) for _ in imgs_info]


class TestOVDetectionModelTiling:
    """Tests for tile-aware inference in OVDetectionModel (forward_tiles / test_step routing)."""

    @pytest.fixture
    def detection_model(self):
        with patch.object(OVDetectionModel, "__init__", lambda *_args, **_kwargs: None):
            model = OVDetectionModel.__new__(OVDetectionModel)
            model.model = MagicMock()
            model.tile_config = TileConfig(enable_tiler=True, tile_size=(200, 200))
            model._label_info = MagicMock()
            model._label_info.num_classes = 3
            model.hparams = {}
            return model

    def _make_tile_batch(self) -> _FakeTileBatchDet:
        ori_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(256, 256),
            ori_shape=(256, 256),
        )
        # One original image expanded into two native-resolution tile crops.
        batch_tiles = [[torch.rand(3, 200, 200), torch.rand(3, 200, 200)]]
        tile_infos = [
            _FakeTileInfo(x=0, y=0, width=200, height=200),
            _FakeTileInfo(x=56, y=56, width=200, height=200),
        ]
        return _FakeTileBatchDet(
            imgs_info=[ori_info],
            batch_tiles=batch_tiles,
            batch_tile_tile_infos=[tile_infos],
        )

    def test_forward_tiles_delegates_to_model_api_tiler(self, detection_model):
        """forward_tiles feeds native tiles + coords to the ModelAPI tiler and converts the merged result."""
        tile_batch = self._make_tile_batch()

        # A ModelAPI DetectionResult (numpy) returned by the tiler's predict_tiles merge.
        merged = _FakeDetectionResult(
            bboxes=np.array([[10, 10, 50, 50]], dtype=np.int32),
            scores=np.array([0.9], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        tiler = MagicMock()
        tiler.model.get_label_name.return_value = "object"  # no "background" label shift
        tiler.predict_tiles.return_value = merged
        detection_model._get_tiler = MagicMock(return_value=tiler)

        result = detection_model.forward_tiles(tile_batch)

        # predict_tiles received the two native tiles (HWC numpy) and their absolute coords.
        tiler.predict_tiles.assert_called_once()
        call_tiles, call_coords, call_shape = tiler.predict_tiles.call_args.args
        assert len(call_tiles) == 2
        assert call_tiles[0].shape == (200, 200, 3)
        assert call_coords == [[0, 0, 200, 200], [56, 56, 256, 256]]
        assert call_shape == (256, 256, 3)

        # The merged ModelAPI result is converted to a getitune PredictionBatch.
        assert isinstance(result, PredictionBatch)
        torch.testing.assert_close(
            result.bboxes[0].data,  # pyrefly: ignore[unsupported-operation]
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        )
        assert result.bboxes[0].canvas_size == (256, 256)  # pyrefly: ignore[unsupported-operation]
        torch.testing.assert_close(result.scores[0], torch.tensor([0.9]))  # pyrefly: ignore[unsupported-operation]
        torch.testing.assert_close(result.labels[0], torch.tensor([1]))  # pyrefly: ignore[unsupported-operation]

    def test_forward_tiles_applies_background_label_shift(self, detection_model):
        """A leading 'background' label must shift merged labels down by one."""
        tile_batch = self._make_tile_batch()
        merged = _FakeDetectionResult(
            bboxes=np.array([[10, 10, 50, 50]], dtype=np.int32),
            scores=np.array([0.9], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        tiler = MagicMock()
        tiler.model.get_label_name.return_value = "background"  # triggers label shift
        tiler.predict_tiles.return_value = merged
        detection_model._get_tiler = MagicMock(return_value=tiler)

        result = detection_model.forward_tiles(tile_batch)
        torch.testing.assert_close(result.labels[0], torch.tensor([0]))  # pyrefly: ignore[unsupported-operation]

    def test_test_step_routes_tile_batch_to_forward_tiles(self, detection_model):
        """test_step must dispatch TileBatchData inputs to forward_tiles, not the plain forward path."""
        tile_batch = self._make_tile_batch()
        merged_preds = PredictionBatch(
            images=[torch.empty(3, 256, 256)],
            imgs_info=[tile_batch.imgs_info[0]],
            scores=[torch.tensor([0.9])],
            bboxes=[
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    torch.tensor([[1.0, 2.0, 3.0, 4.0]]), format="XYXY", canvas_size=(256, 256)
                )
            ],
            labels=[torch.tensor([0], dtype=torch.long)],
        )
        detection_model.forward_tiles = MagicMock(return_value=merged_preds)
        detection_model.forward = MagicMock()
        metric = MagicMock()

        detection_model.test_step(tile_batch, metric)

        detection_model.forward_tiles.assert_called_once_with(tile_batch)
        detection_model.forward.assert_not_called()
        metric.update.assert_called_once()

    def test_get_tiler_reuses_existing_model_api_tiler(self, detection_model):
        """_get_tiler returns the already-wrapped ModelAPI tiler without constructing a new one."""
        from model_api.tilers import DetectionTiler

        existing = DetectionTiler.__new__(DetectionTiler)
        detection_model.model = existing
        assert detection_model._get_tiler() is existing

    def test_forward_untiled_bypasses_model_api_tiler(self, detection_model):
        """_forward_untiled must infer on the raw model, never the model_api Tiler wrapper.

        Regression guard for the tiled-evaluation double-tiling bug: the datamodule
        already produces the tiles, so routing them through the model_api ``Tiler``
        (installed by ``_setup_tiler`` for tiling-aware IR models) would tile each
        tile a second time, causing a combinatorial explosion of inference calls
        and effectively hanging evaluation.
        """
        from model_api.tilers import DetectionTiler

        raw_model = MagicMock(name="raw_model")
        raw_model.infer_batch.return_value = [_FakeDetectionResult(np.empty((0, 4)), np.empty((0,)), np.empty((0,)))]

        # Wrap the raw model in a real DetectionTiler instance (attribute access only,
        # no inference), mirroring the state after ``_setup_tiler``.
        tiler = DetectionTiler.__new__(DetectionTiler)
        tiler.model = raw_model
        detection_model.model = tiler

        sample_batch = SampleBatch(images=[torch.rand(3, 200, 200)], imgs_info=[self._make_tile_batch().imgs_info[0]])
        detection_model._customize_inputs = MagicMock(return_value={"inputs": [np.empty((200, 200, 3))]})
        detection_model._customize_outputs = MagicMock(return_value=PredictionBatch(images=[], imgs_info=[]))

        detection_model._forward_untiled(sample_batch)

        # Inference ran on the underlying (untiled) model, not on the Tiler wrapper.
        raw_model.infer_batch.assert_called_once()
