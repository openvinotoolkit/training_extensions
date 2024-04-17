from unittest.mock import Mock, patch

import numpy as np
import pytest
from model_api.models.utils import (
    ClassificationResult,
    Detection,
    DetectionResult,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    SegmentedObject,
)
from numpy.random import PCG64, Generator


@pytest.fixture(scope="module", autouse=True)
def fxt_import_module():
    global BaseVisualizer, ClassificationVisualizer, InstanceSegmentationVisualizer, ObjectDetectionVisualizer, SemanticSegmentationVisualizer  # noqa: PLW0603
    from otx.core.exporter.exportable_code.demo.demo_package import (
        BaseVisualizer as _BaseVisualizer,
    )
    from otx.core.exporter.exportable_code.demo.demo_package import (
        ClassificationVisualizer as _ClassificationVisualizer,
    )
    from otx.core.exporter.exportable_code.demo.demo_package import (
        InstanceSegmentationVisualizer as _InstanceSegmentationVisualizer,
    )
    from otx.core.exporter.exportable_code.demo.demo_package import (
        ObjectDetectionVisualizer as _ObjectDetectionVisualizer,
    )
    from otx.core.exporter.exportable_code.demo.demo_package import (
        SemanticSegmentationVisualizer as _SemanticSegmentationVisualizer,
    )

    BaseVisualizer = _BaseVisualizer
    ClassificationVisualizer = _ClassificationVisualizer
    InstanceSegmentationVisualizer = _InstanceSegmentationVisualizer
    ObjectDetectionVisualizer = _ObjectDetectionVisualizer
    SemanticSegmentationVisualizer = _SemanticSegmentationVisualizer


class TestBaseVisualizer:
    def test_init(self):
        visualizer = BaseVisualizer(window_name="TestWindow", no_show=True, delay=10, output="test_output")
        assert visualizer.window_name == "TestWindow"
        assert visualizer.no_show is True
        assert visualizer.delay == 10
        assert visualizer.output == "test_output"

    # Test show method without displaying the window
    @patch("cv2.imshow")
    def test_show_no_display(self, mock_imshow):
        visualizer = BaseVisualizer(no_show=True)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        visualizer.show(test_image)
        mock_imshow.assert_not_called()

    # Test show method with displaying the window
    @patch("cv2.imshow")
    def test_show_display(self, mock_imshow):
        visualizer = BaseVisualizer(no_show=False)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        visualizer.show(test_image)
        mock_imshow.assert_called_once_with(visualizer.window_name, test_image)

    # Test is_quit method
    @patch("cv2.waitKey", return_value=ord("q"))
    def test_is_quit(self, mock_waitkey):
        visualizer = BaseVisualizer(no_show=False)
        assert visualizer.is_quit() is True

    # Test video_delay method
    @patch("time.sleep")
    def test_video_delay(self, mock_sleep):
        streamer = Mock()
        streamer.get_type.return_value = "VIDEO"
        streamer.fps.return_value = 30
        visualizer = BaseVisualizer(no_show=False)
        visualizer.video_delay(0.02, streamer)
        mock_sleep.assert_called_once_with(1 / 30 - 0.02)


class TestClassificationVisualizer:
    @pytest.fixture()
    def visualizer(self):
        return ClassificationVisualizer(window_name="TestWindow", no_show=True, delay=10, output="test_output")

    @pytest.fixture()
    def frame(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @pytest.fixture()
    def predictions(self):
        return ClassificationResult(
            top_labels=[(0, "cat", 0.9)],
            saliency_map=None,
            feature_vector=None,
            raw_scores=[0.9],
        )

    def test_draw_one_prediction(self, frame, predictions, visualizer):
        # test one prediction
        copied_frame = frame.copy()
        output = visualizer.draw(frame, predictions)
        assert output.shape == (100, 100, 3)
        assert np.any(output != copied_frame)

    def test_draw_multiple_predictions(self, frame, predictions, visualizer):
        # test multiple predictions
        copied_frame = frame.copy()
        predictions.top_labels.extend([(1, "dog", 0.8), (2, "bird", 0.7)])
        output = visualizer.draw(frame, predictions)
        assert output.shape == (100, 100, 3)
        assert np.any(output != copied_frame)

    def test_label_overflow(self, frame, predictions, visualizer):
        # test multiple predictions
        copied_frame = frame.copy()
        predictions.top_labels.extend([(1, "dog", 0.8), (2, "bird", 0.7), (3, "cat", 0.6)])
        output = visualizer.draw(frame, predictions)
        assert output.shape == (100, 100, 3)
        assert np.any(output != copied_frame)

    def test_draw_no_predictions(self, frame, visualizer):
        # test no predictions
        copied_frame = frame.copy()
        predictions = ClassificationResult(top_labels=[()], saliency_map=None, feature_vector=None, raw_scores=[])
        output = visualizer.draw(frame, predictions)
        assert output.shape == (100, 100, 3)
        assert np.equal(output, copied_frame).all()


class TestDetectionVisualizer:
    @pytest.fixture()
    def visualizer(self):
        return ObjectDetectionVisualizer(
            labels=["Pedestrian", "Car"],
            window_name="TestWindow",
            no_show=True,
            delay=10,
            output="test_output",
        )

    def test_draw_no_predictions(self, visualizer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult([], saliency_map=None, feature_vector=None)
        output_frame = visualizer.draw(frame, predictions)
        assert np.array_equal(frame, output_frame)

    def test_draw_with_predictions(self, visualizer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = DetectionResult(
            [Detection(10, 40, 30, 80, 0.7, 2, "Car")],
            saliency_map=None,
            feature_vector=None,
        )
        copied_frame = frame.copy()
        output_frame = visualizer.draw(frame, predictions)
        assert np.any(output_frame != copied_frame)


class TestInstanceSegmentationVisualizer:
    @pytest.fixture()
    def rand_generator(self):
        return Generator(PCG64())

    @pytest.fixture()
    def visualizer(self):
        return InstanceSegmentationVisualizer(
            labels=["person", "car"],
            window_name="TestWindow",
            no_show=True,
            delay=10,
            output="test_output",
        )

    def test_draw_multiple_objects(self, visualizer, rand_generator):
        # Create a frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        copied_frame = frame.copy()

        # Create instance segmentation results with multiple objects
        predictions = InstanceSegmentationResult(
            segmentedObjects=[
                SegmentedObject(
                    xmin=10,
                    ymin=10,
                    xmax=30,
                    ymax=30,
                    score=0.9,
                    id=0,
                    mask=rand_generator.integers(2, size=(100, 100), dtype=np.uint8),
                    str_label="person",
                ),
                SegmentedObject(
                    xmin=40,
                    ymin=40,
                    xmax=60,
                    ymax=60,
                    score=0.8,
                    id=1,
                    mask=rand_generator.integers(2, size=(100, 100), dtype=np.uint8),
                    str_label="car",
                ),
            ],
            saliency_map=None,
            feature_vector=None,
        )

        drawn_frame = visualizer.draw(frame, predictions)
        assert np.any(drawn_frame != copied_frame)

        # Assertion checks for the drawn frame

    def test_draw_no_objects(self, visualizer):
        # Create a frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        copied_frame = frame.copy()

        # Create instance segmentation results with no objects
        predictions = InstanceSegmentationResult(segmentedObjects=[], saliency_map=None, feature_vector=None)

        drawn_frame = visualizer.draw(frame, predictions)
        assert np.array_equal(drawn_frame, copied_frame)


class TestSemanticSegmentationVisualizer:
    @pytest.fixture()
    def labels(self):
        return ["background", "object1", "object2"]

    @pytest.fixture()
    def visualizer(self, labels):
        return SemanticSegmentationVisualizer(
            labels=labels,
            window_name="TestWindow",
            no_show=True,
            delay=10,
            output="test_output",
        )

    @pytest.fixture()
    def rand_generator(self):
        return Generator(PCG64())

    def test_initialization(self, visualizer):
        assert isinstance(visualizer.color_palette, np.ndarray)
        assert visualizer.color_map.shape == (256, 1, 3)
        assert visualizer.color_map.dtype == np.uint8

    def test_create_color_map(self, visualizer):
        color_map = visualizer._create_color_map()
        assert color_map.shape == (256, 1, 3)
        assert color_map.dtype == np.uint8

    def test_apply_color_map(self, visualizer, labels, rand_generator):
        input_2d_mask = rand_generator.integers(0, len(labels), size=(10, 10))
        colored_mask = visualizer._apply_color_map(input_2d_mask)
        assert colored_mask.shape == (10, 10, 3)

    def test_draw(self, visualizer, rand_generator):
        frame = rand_generator.integers(0, 255, size=(10, 10, 3), dtype=np.uint8)
        copied_frame = frame.copy()
        masks = ImageResultWithSoftPrediction(
            resultImage=rand_generator.integers(0, 255, size=(10, 10), dtype=np.uint8),
            soft_prediction=rand_generator.random((10, 10)),
            saliency_map=None,
            feature_vector=None,
        )
        output_image = visualizer.draw(frame, masks)
        assert output_image.shape == frame.shape
        assert np.any(output_image != copied_frame)
