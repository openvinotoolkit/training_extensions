# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import pytest
from model_api.models import ClassificationResult, DetectionResult, InstanceSegmentationResult
from model_api.models.result import Label
from model_api.visualizer import BoundingBox, Flatten, Polygon
from model_api.visualizer import Label as VisualizerLabel

from app.utils.visualization import (
    ClassificationVisualizerCreator,
    DetectionVisualizerCreator,
    InstanceSegmentationVisualizerCreator,
    VisualizationDispatcher,
    Visualizer,
    _compute_scale,
)

bboxes = np.array([[10, 20, 50, 60], [30, 40, 70, 80], [15, 25, 55, 65]], dtype=np.int32)

labels = np.array([1, 2, 3], dtype=np.int32)

masks = np.array(
    [
        np.zeros((100, 100), dtype=np.uint8),
        np.ones((100, 100), dtype=np.uint8),
        np.full((100, 100), 255, dtype=np.uint8),
    ],
    dtype=np.uint8,
)


class TestVisualizationDispatcherValidation(unittest.TestCase):
    def test_handles_empty_image_input(self):
        dispatcher = VisualizationDispatcher()
        original_image = np.array([])
        predictions = DetectionResult(bboxes, labels)
        with self.assertRaises(Exception):
            dispatcher.create_visualization(original_image, predictions)


class TestDetectionVisualizerCreator(unittest.TestCase):
    def test_creates_visualization(self):
        creator = DetectionVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))

    def test_uses_visualizer_label_in_layout(self):
        creator = DetectionVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels)
        with mock.patch("model_api.visualizer.scene.DetectionScene") as mock_scene:
            creator.create_visualization(original_image, predictions)
        layout = mock_scene.call_args.kwargs["layout"]
        self.assertIsInstance(layout, Flatten)
        self.assertEqual(layout.children, (BoundingBox, VisualizerLabel))


class TestInstanceSegmentationVisualizerCreator(unittest.TestCase):
    def test_creates_visualization(self):
        creator = InstanceSegmentationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = InstanceSegmentationResult(bboxes, labels, masks)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))

    def test_uses_visualizer_label_in_layout(self):
        creator = InstanceSegmentationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = InstanceSegmentationResult(bboxes, labels, masks)
        with mock.patch("model_api.visualizer.scene.InstanceSegmentationScene") as mock_scene:
            creator.create_visualization(original_image, predictions)
        layout = mock_scene.call_args.kwargs["layout"]
        self.assertIsInstance(layout, Flatten)
        self.assertEqual(layout.children, (Polygon, VisualizerLabel))


class TestClassificationVisualizerCreator(unittest.TestCase):
    def test_creates_visualization(self):
        creator = ClassificationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        classification_labels = [Label(id=1, name="1", confidence=0.9), Label(id=2, name="2", confidence=0.1)]
        predictions = ClassificationResult(classification_labels)
        result = creator.create_visualization(original_image, predictions)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.array_equal(result, original_image))


class TestVisualizationHelpers(unittest.TestCase):
    def test_compute_scale_handles_none_or_empty(self):
        assert _compute_scale(None) == 1.0  # type: ignore[arg-type]
        assert _compute_scale(np.array([])) == 1.0

    def test_compute_scale_never_below_one(self):
        small = np.zeros((100, 200, 3), dtype=np.uint8)
        assert _compute_scale(small) == 1.0

    def test_compute_scale_scales_with_longer_edge(self):
        # 4K longer edge → ~3.0
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        assert _compute_scale(img) == pytest.approx(3.0, rel=1e-3)


class TestLabelColors(unittest.TestCase):
    """The project label colors must be forwarded to Model API so that the rendered
    predictions use the same colors as the labels defined in the project."""

    LABEL_COLORS = {"1": "#00FF00", "2": "#FF0000", "3": "#0000FF"}

    def test_detection_creator_forwards_label_colors(self):
        creator = DetectionVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels)
        with mock.patch("model_api.visualizer.scene.DetectionScene") as mock_scene:
            creator.create_visualization(original_image, predictions, self.LABEL_COLORS)
        self.assertEqual(mock_scene.call_args.kwargs["label_colors"], self.LABEL_COLORS)

    def test_instance_segmentation_creator_forwards_label_colors(self):
        creator = InstanceSegmentationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = InstanceSegmentationResult(bboxes, labels, masks)
        with mock.patch("model_api.visualizer.scene.InstanceSegmentationScene") as mock_scene:
            creator.create_visualization(original_image, predictions, self.LABEL_COLORS)
        self.assertEqual(mock_scene.call_args.kwargs["label_colors"], self.LABEL_COLORS)

    def test_classification_creator_forwards_label_colors(self):
        creator = ClassificationVisualizerCreator()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = ClassificationResult([Label(id=1, name="1", confidence=0.9)])
        with mock.patch("model_api.visualizer.scene.ClassificationScene") as mock_scene:
            creator.create_visualization(original_image, predictions, self.LABEL_COLORS)
        self.assertEqual(mock_scene.call_args.kwargs["label_colors"], self.LABEL_COLORS)

    def test_dispatcher_forwards_label_colors(self):
        dispatcher = VisualizationDispatcher()
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels)
        creator = mock.Mock()
        with mock.patch.dict(dispatcher._creator_map, {DetectionResult: creator}):
            dispatcher.create_visualization(original_image, predictions, self.LABEL_COLORS)
        creator.create_visualization.assert_called_once_with(original_image, predictions, self.LABEL_COLORS)

    def test_overlay_predictions_forwards_label_colors(self):
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels)
        with mock.patch.object(VisualizationDispatcher, "create_visualization", return_value=original_image) as mocked:
            Visualizer.overlay_predictions(original_image, predictions, label_colors=self.LABEL_COLORS)
        self.assertEqual(mocked.call_args.kwargs["label_colors"], self.LABEL_COLORS)

    def test_rendered_detection_uses_the_requested_color(self):
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(bboxes, labels, label_names=["cat", "dog", "fish"])
        creator = DetectionVisualizerCreator()

        default_render = creator.create_visualization(original_image, predictions)
        custom_render = creator.create_visualization(original_image, predictions, {"cat": (255, 0, 255)})

        self.assertFalse(np.array_equal(default_render, custom_render))
        # The requested color must actually be drawn on the image
        self.assertTrue(np.any(np.all(custom_render == np.array([255, 0, 255], dtype=np.uint8), axis=-1)))
