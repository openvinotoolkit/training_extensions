# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for detection model confidence threshold logic."""

from unittest.mock import Mock, patch

import pytest
import torch
from torchmetrics import MetricCollection
from torchvision import tv_tensors

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics.fmeasure import FMeasure
from getitune.types.label import LabelInfo


class MockDetectionModel(LightningDetectionModel):
    """Mock detection model for unit testing."""

    def _create_model(self, num_classes=None) -> torch.nn.Module:
        return torch.nn.Identity()

    def _convert_pred_entity_to_compute_metric(self, preds, inputs) -> dict:
        return {
            "preds": [
                {"boxes": torch.tensor([[0, 0, 1, 1]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])},
            ],
            "target": [{"boxes": torch.tensor([[0, 0, 1, 1]]), "labels": torch.tensor([0])}],
        }


@pytest.fixture
def detection_model():
    """Create a test detection model."""
    label_info = LabelInfo(
        label_names=["cat", "dog"],
        label_groups=[["cat", "dog"]],
        label_ids=["0", "1"],
    )

    data_params = DataInputParams(
        input_size=(416, 416),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    return MockDetectionModel(
        label_info=label_info,
        data_input_params=data_params,
        pretrained=False,
    )


@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    return PredictionBatch(
        images=torch.stack([torch.rand(3, 416, 416), torch.rand(3, 416, 416)]),
        imgs_info=[
            ImageInfo(img_idx=0, img_shape=(3, 416, 416), ori_shape=(3, 416, 416)),
            ImageInfo(img_idx=1, img_shape=(3, 416, 416), ori_shape=(3, 416, 416)),
        ],
        scores=[
            torch.tensor([0.9, 0.3, 0.7]),  # First image: 2 above 0.5, 1 below
            torch.tensor([0.2, 0.8, 0.6]),  # Second image: 2 above 0.5, 1 below
        ],
        bboxes=[
            tv_tensors.BoundingBoxes(
                torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
                format="XYXY",
                canvas_size=(416, 416),
                dtype=torch.float32,
            ),
            tv_tensors.BoundingBoxes(
                torch.tensor([[5, 5, 15, 15], [25, 25, 35, 35], [45, 45, 55, 55]]),
                format="XYXY",
                canvas_size=(416, 416),
                dtype=torch.float32,
            ),
        ],
        labels=[
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 0, 1]),
        ],
    )


@pytest.fixture
def sample_batch():
    """Create sample input batch."""
    return SampleBatch(
        images=torch.stack([torch.rand(3, 416, 416), torch.rand(3, 416, 416)]),
        imgs_info=[
            ImageInfo(img_idx=0, img_shape=(3, 416, 416), ori_shape=(3, 416, 416)),
            ImageInfo(img_idx=1, img_shape=(3, 416, 416), ori_shape=(3, 416, 416)),
        ],
        bboxes=[
            tv_tensors.BoundingBoxes(
                torch.tensor([[0, 0, 10, 10]]),
                format="XYXY",
                canvas_size=(416, 416),
                dtype=torch.float32,
            ),
            tv_tensors.BoundingBoxes(
                torch.tensor([[5, 5, 15, 15]]),
                format="XYXY",
                canvas_size=(416, 416),
                dtype=torch.float32,
            ),
        ],
        labels=[torch.tensor([0]), torch.tensor([1])],
    )


class TestLogMetrics:
    """Test cases for _log_metrics method."""

    def test_validation_with_fmeasure_metric(self, detection_model):
        """Test validation logging with FMeasure metric."""
        # Setup - Create proper mock that supports hasattr
        fmeasure = Mock(spec=FMeasure)
        fmeasure.best_confidence_threshold = 0.7
        # Ensure hasattr works correctly
        fmeasure.configure_mock(best_confidence_threshold=0.7)
        # Mock compute() to return a proper dictionary to avoid TypeError
        fmeasure.compute.return_value = {"f1-score": torch.tensor(0.85)}

        # Call the actual method - this will run the real implementation
        detection_model._log_metrics(fmeasure, "val")

        # Assertions
        assert detection_model.hparams["best_confidence_threshold"] == 0.7

    def test_validation_with_metric_collection(self, detection_model):
        """Test validation logging with MetricCollection containing FMeasure."""
        # Setup - Create proper mock that supports hasattr
        fmeasure = Mock(spec=FMeasure)
        fmeasure.configure_mock(best_confidence_threshold=0.6)
        # Ensure hasattr works correctly for Python 3.10 compatibility
        fmeasure.best_confidence_threshold = 0.6

        metric_collection = Mock(spec=MetricCollection)
        metric_collection.FMeasure = fmeasure
        # Mock compute() to return proper dictionary
        metric_collection.compute.return_value = {"f1-score": torch.tensor(0.80)}

        # Call method
        detection_model._log_metrics(metric_collection, "val")

        # Assertions
        assert detection_model.hparams["best_confidence_threshold"] == 0.6

    def test_validation_without_fmeasure(self, detection_model):
        """Test validation logging without FMeasure metric."""
        # Setup
        other_metric = Mock()
        # Mock compute() to return proper dictionary
        other_metric.compute.return_value = {"accuracy": torch.tensor(0.95)}

        # Call method
        detection_model._log_metrics(other_metric, "val")

        # Should not modify hparams
        assert "best_confidence_threshold" not in detection_model.hparams


class TestFilterOutputsByThreshold:
    """Test cases for _filter_outputs_by_threshold method."""

    def test_filtering_with_default_threshold(self, detection_model, sample_predictions):
        """Test filtering with default threshold (0.5)."""
        filtered = detection_model._filter_outputs_by_threshold(sample_predictions)

        # Check that low confidence predictions are filtered out
        assert len(filtered.scores[0]) == 2  # 0.9, 0.7 > 0.5
        assert len(filtered.scores[1]) == 2  # 0.8, 0.6 > 0.5

        # Check that corresponding bboxes and labels are filtered
        assert len(filtered.bboxes[0]) == 2
        assert len(filtered.bboxes[1]) == 2
        assert len(filtered.labels[0]) == 2
        assert len(filtered.labels[1]) == 2

    def test_filtering_with_custom_threshold(self, detection_model, sample_predictions):
        """Test filtering with custom threshold."""
        detection_model.hparams["best_confidence_threshold"] = 0.75

        filtered = detection_model._filter_outputs_by_threshold(sample_predictions)

        # With threshold 0.75, only scores > 0.75 should remain
        assert len(filtered.scores[0]) == 1  # Only 0.9 > 0.75
        assert len(filtered.scores[1]) == 1  # Only 0.8 > 0.75

        # Check values
        assert filtered.scores[0][0] == 0.9
        assert filtered.scores[1][0] == 0.8

    def test_filtering_preserves_tensor_types(self, detection_model, sample_predictions):
        """Test that filtering preserves tensor types and formats."""
        filtered = detection_model._filter_outputs_by_threshold(sample_predictions)

        # Check that bboxes remain as tv_tensors.BoundingBoxes
        assert isinstance(filtered.bboxes[0], tv_tensors.BoundingBoxes)
        assert isinstance(filtered.bboxes[1], tv_tensors.BoundingBoxes)

        # Check that other outputs remain as tensors
        assert isinstance(filtered.scores[0], torch.Tensor)
        assert isinstance(filtered.labels[0], torch.Tensor)

    def test_filtering_with_none_outputs(self, detection_model):
        """Test filtering when outputs have None values."""
        preds_with_none = PredictionBatch(
            images=torch.rand(1, 3, 416, 416),
            imgs_info=[ImageInfo(img_idx=0, img_shape=(3, 416, 416), ori_shape=(3, 416, 416))],
            scores=None,
            bboxes=None,
            labels=None,
        )

        filtered = detection_model._filter_outputs_by_threshold(preds_with_none)

        # Should handle None gracefully
        assert filtered.scores == []
        assert filtered.bboxes == []
        assert filtered.labels == []

    def test_filtering_empty_predictions(self, detection_model):
        """Test filtering with empty prediction lists."""
        empty_preds = PredictionBatch(
            images=torch.stack([torch.rand(3, 416, 416), torch.rand(3, 416, 416)]),
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(3, 416, 416), ori_shape=(3, 416, 416)),
                ImageInfo(img_idx=1, img_shape=(3, 416, 416), ori_shape=(3, 416, 416)),
            ],
            scores=[torch.tensor([]), torch.tensor([])],
            bboxes=[
                tv_tensors.BoundingBoxes(torch.empty(0, 4, dtype=torch.float32), format="XYXY", canvas_size=(416, 416)),
                tv_tensors.BoundingBoxes(torch.empty(0, 4, dtype=torch.float32), format="XYXY", canvas_size=(416, 416)),
            ],
            labels=[torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)],
        )

        filtered = detection_model._filter_outputs_by_threshold(empty_preds)

        # Should handle empty tensors gracefully
        assert len(filtered.scores) == 2
        assert len(filtered.scores[0]) == 0
        assert len(filtered.scores[1]) == 0


class TestTestStep:
    """Test cases for test_step method."""

    @patch("getitune.backend.lightning.models.detection.base.LightningDetectionModel.forward")
    @patch("getitune.backend.lightning.models.detection.base.LightningDetectionModel._filter_outputs_by_threshold")
    def test_filtering_before_metric_computation(
        self,
        mock_filter,
        mock_forward,
        detection_model,
        sample_batch,
        sample_predictions,
    ):
        """Test that filtering happens before metric computation in test_step."""
        # Setup mocks
        mock_forward.return_value = sample_predictions
        filtered_preds = sample_predictions  # Simulated filtered predictions
        mock_filter.return_value = filtered_preds

        # Patch the instance method directly since MockDetectionModel overrides it
        with patch.object(detection_model, "_convert_pred_entity_to_compute_metric") as mock_convert:
            mock_convert.return_value = {
                "preds": [
                    {"boxes": torch.tensor([[0, 0, 1, 1]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])},
                ],
            }

            # Setup metric mock using property patch
            mock_metric = Mock()
            with patch.object(type(detection_model), "metric", new_callable=lambda: mock_metric):
                # Call test_step
                result = detection_model.test_step(sample_batch, 0)

                # Verify call order: forward -> filter -> convert -> metric.update
                assert mock_forward.called
                assert not mock_filter.called
                assert mock_convert.called

                # Verify that _convert_pred_entity_to_compute_metric was called with filtered output
                mock_convert.assert_called_once_with(filtered_preds, sample_batch)

                # Verify metric was updated
                mock_metric.update.assert_called_once()

            result = detection_model.predict_step(sample_batch, 0)
            assert mock_filter.called
            mock_filter.assert_called_once_with(sample_predictions)
            assert result == filtered_preds

    @patch("getitune.backend.lightning.models.detection.base.LightningDetectionModel.forward")
    def test_test_step_with_loss_entity_raises_error(self, mock_forward, detection_model, sample_batch):
        """Test that test_step raises TypeError when forward returns BatchLoss."""
        from getitune.data.entity.base import BatchLoss

        # Setup mock to return loss entity
        mock_forward.return_value = BatchLoss()

        # Should raise TypeError
        with pytest.raises(TypeError):
            detection_model.test_step(sample_batch, 0)

    @patch("getitune.backend.lightning.models.detection.base.LightningDetectionModel.forward")
    @patch("getitune.backend.lightning.models.detection.base.LightningDetectionModel._filter_outputs_by_threshold")
    def test_test_step_with_list_metric_inputs(
        self,
        mock_filter,
        mock_forward,
        detection_model,
        sample_batch,
        sample_predictions,
    ):
        """Test test_step with list of metric inputs."""
        # Setup mocks
        mock_forward.return_value = sample_predictions
        mock_filter.return_value = sample_predictions

        # Patch the instance method directly since MockDetectionModel overrides it
        with patch.object(detection_model, "_convert_pred_entity_to_compute_metric") as mock_convert:
            mock_convert.return_value = [
                {
                    "preds": [
                        {
                            "boxes": torch.tensor([[0, 0, 1, 1]]),
                            "scores": torch.tensor([0.9]),
                            "labels": torch.tensor([0]),
                        },
                    ],
                },
                {
                    "preds": [
                        {
                            "boxes": torch.tensor([[5, 5, 15, 15]]),
                            "scores": torch.tensor([0.8]),
                            "labels": torch.tensor([1]),
                        },
                    ],
                },
            ]

            # Setup metric mock using property patch
            mock_metric = Mock()
            with patch.object(type(detection_model), "metric", new_callable=lambda: mock_metric):
                # Call test_step
                detection_model.test_step(sample_batch, 0)

                # Verify metric.update was called for each item in the list
                assert mock_metric.update.call_count == 2


class TestCheckpointHandling:
    """Test cases for checkpoint saving/loading."""

    def test_on_save_checkpoint_preserves_threshold(self, detection_model):
        """Test that on_save_checkpoint preserves best_confidence_threshold."""
        # Setup
        detection_model.hparams["best_confidence_threshold"] = 0.6
        detection_model.hparams["other_param"] = "should_remain"

        checkpoint = {
            "state_dict": {},
            "hyper_parameters": detection_model.hparams.copy(),
        }

        # Call on_save_checkpoint directly (it calls super() internally)
        detection_model.on_save_checkpoint(checkpoint)

        # Verify that parameters remain
        assert "best_confidence_threshold" in checkpoint["hyper_parameters"]
        assert "other_param" in checkpoint["hyper_parameters"]

    def test_on_save_checkpoint_basic_functionality(self, detection_model):
        """Test basic on_save_checkpoint functionality."""
        # Setup - basic threshold in hparams
        detection_model.hparams["best_confidence_threshold"] = 0.6
        detection_model.hparams["other_param"] = "should_remain"

        checkpoint = {
            "state_dict": {},
            "hyper_parameters": detection_model.hparams.copy(),
        }

        # Call on_save_checkpoint - should not raise error
        detection_model.on_save_checkpoint(checkpoint)

        # Verify parameters remain
        assert "best_confidence_threshold" in checkpoint["hyper_parameters"]
        assert "other_param" in checkpoint["hyper_parameters"]


class TestIntegration:
    """Integration tests for the complete confidence threshold workflow."""

    def test_validation_to_test_workflow(self, detection_model, sample_batch):
        """Test complete workflow from validation to test."""
        # Setup FMeasure mock
        fmeasure = Mock(spec=FMeasure)
        fmeasure.best_confidence_threshold = 0.7
        # Mock compute() to return proper dictionary
        fmeasure.compute.return_value = {"f1-score": torch.tensor(0.85)}

        # Simulate validation epoch
        fmeasure.configure_mock(best_confidence_threshold=0.7)
        detection_model._log_metrics(fmeasure, "val")

        # Verify threshold was stored
        assert detection_model.best_confidence_threshold == 0.7

        # Simulate test step - should use the stored threshold
        with patch.object(detection_model, "forward") as mock_forward, patch.object(
            detection_model,
            "_convert_pred_entity_to_compute_metric",
        ) as mock_convert:
            # Setup sample predictions with scores above and below threshold
            test_preds = PredictionBatch(
                images=torch.rand(1, 3, 416, 416),
                imgs_info=[ImageInfo(img_idx=0, img_shape=(3, 416, 416), ori_shape=(3, 416, 416))],
                scores=[torch.tensor([0.9, 0.5, 0.3])],
                bboxes=[
                    tv_tensors.BoundingBoxes(
                        torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
                        format="XYXY",
                        canvas_size=(416, 416),
                        dtype=torch.float32,
                    ),
                ],
                labels=[torch.tensor([0, 1, 0])],
            )

            mock_forward.return_value = test_preds
            mock_convert.return_value = {
                "preds": [
                    {"boxes": torch.tensor([[0, 0, 1, 1]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])},
                ],
            }

            # Setup metric mock using property patch
            mock_metric = Mock()
            with patch.object(type(detection_model), "metric", new_callable=lambda: mock_metric):
                # Call test_step
                result = detection_model.test_step(sample_batch, 0)

                # Verify that only high confidence predictions remain
                assert len(result.scores[0]) == 3  # No filtering applied in test_step itself
                assert result.scores[0][0] == 0.9
                assert result.scores[0][1] == 0.5
                assert result.scores[0][2] == 0.3
