# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RF-DETR detection model."""

from __future__ import annotations

import logging

import pytest
import torch
from torchvision import tv_tensors

from getitune.backend.lightning.models.detection.rfdetr import RFDETR
from getitune.backend.lightning.models.detection.utils.rfdetr_batch_utils import (
    _cap_per_image,
    _get_largest_indices,
    _proportional_limit,
    _reduce_dense_images,
    _subset_target,
    limit_batch_objects,
)
from getitune.data.entity import PredictionBatch
from getitune.data.entity.sample import SampleBatch


class TestRFDETRBatchLimitingUtils:
    """Test class for RF-DETR batch object limiting utility functions."""

    def _create_targets(self, object_counts: list[int]) -> list[dict[str, torch.Tensor]]:
        """Create mock targets with specified object counts.

        Args:
            object_counts: List of object counts per image.

        Returns:
            List of target dicts with boxes and labels.
        """
        targets = []
        for count in object_counts:
            # Create boxes with varying sizes for area-based selection
            # Format: cxcywh normalized
            boxes = torch.rand(count, 4) * 0.5 + 0.1  # Ensure reasonable sizes
            labels = torch.randint(0, 10, (count,))
            targets.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "size": torch.tensor([512, 512]),
                    "orig_size": torch.tensor([512, 512]),
                }
            )
        return targets

    def test_no_limiting_when_under_budget(self) -> None:
        """Test that no limiting happens when total objects are under budget."""
        targets = self._create_targets([100, 100, 100, 100])  # 400 total < 600
        limited = limit_batch_objects(targets, max_total=600)

        total_after = sum(len(t["boxes"]) for t in limited)
        assert total_after == 400, "Should not limit when under budget"

    def test_per_image_cap_only(self) -> None:
        """Test that per-image cap (300) is applied first."""
        # 1 image with 500 objects, others sparse - total 700 > 600
        targets = self._create_targets([500, 50, 50, 100])  # 700 total
        limited = limit_batch_objects(targets, max_total=600, max_per_image=300)

        # Dense image should be capped to 300 first, bringing total to 500 <= 600
        assert len(limited[0]["boxes"]) == 300
        # Sparse images preserved
        assert len(limited[1]["boxes"]) == 50
        assert len(limited[2]["boxes"]) == 50
        assert len(limited[3]["boxes"]) == 100

    def test_dense_images_reduced_sparse_preserved(self) -> None:
        """Test that dense images are reduced while sparse are preserved."""
        # 2 sparse + 2 dense, total exceeds budget
        targets = self._create_targets([100, 100, 250, 250])  # 700 total
        limited = limit_batch_objects(targets, max_total=600, max_per_image=300)

        total_after = sum(len(t["boxes"]) for t in limited)
        assert total_after <= 600, f"Total {total_after} exceeds limit 600"

        # Sparse images (100 each, threshold=150) should be preserved
        assert len(limited[0]["boxes"]) == 100, "Sparse image 0 should be preserved"
        assert len(limited[1]["boxes"]) == 100, "Sparse image 1 should be preserved"

    def test_proportional_fallback(self) -> None:
        """Test proportional fallback when all images exceed sparse threshold."""
        # All dense images
        targets = self._create_targets([200, 200, 200, 200])  # 800 total
        limited = limit_batch_objects(targets, max_total=400, max_per_image=300)

        total_after = sum(len(t["boxes"]) for t in limited)
        assert total_after <= 400, f"Total {total_after} exceeds limit 400"

    def test_largest_objects_preserved(self) -> None:
        """Test that largest objects by area are preserved."""
        # Create boxes with known areas
        boxes = torch.tensor(
            [
                [0.5, 0.5, 0.1, 0.1],  # area = 0.01 (smallest)
                [0.5, 0.5, 0.2, 0.2],  # area = 0.04
                [0.5, 0.5, 0.3, 0.3],  # area = 0.09
                [0.5, 0.5, 0.4, 0.4],  # area = 0.16 (largest)
            ]
        )
        targets = [
            {
                "boxes": boxes,
                "labels": torch.tensor([0, 1, 2, 3]),
                "size": torch.tensor([512, 512]),
                "orig_size": torch.tensor([512, 512]),
            }
        ]

        # Limit to 2 objects
        limited = limit_batch_objects(targets, max_total=2, max_per_image=2)

        # Should keep boxes with largest areas (indices 2 and 3)
        assert len(limited[0]["boxes"]) == 2
        # Check that largest areas are kept (sorted by index)
        kept_labels = limited[0]["labels"].tolist()
        assert 3 in kept_labels, "Largest box (label=3) should be kept"
        assert 2 in kept_labels, "Second largest box (label=2) should be kept"

    def test_empty_targets(self) -> None:
        """Test handling of empty targets."""
        targets: list[dict] = []
        limited = limit_batch_objects(targets, max_total=600)
        assert limited == []

    def test_empty_boxes_in_target(self) -> None:
        """Test handling of target with empty boxes."""
        targets = [
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
            {"boxes": torch.rand(100, 4), "labels": torch.randint(0, 10, (100,))},
        ]
        limited = limit_batch_objects(targets, max_total=600)

        assert len(limited[0]["boxes"]) == 0
        assert len(limited[1]["boxes"]) == 100

    def test_warning_logged_on_capping(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that warning is logged when capping occurs."""
        targets = self._create_targets([400, 400])  # 800 total > 600

        with caplog.at_level(logging.WARNING):
            limit_batch_objects(targets, max_total=600)

        assert any("RF-DETR batch object limiting" in record.message for record in caplog.records), (
            "Warning should be logged when capping occurs"
        )

    def test_no_warning_when_no_capping(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that no warning is logged when no capping occurs."""
        targets = self._create_targets([100, 100])  # 200 total < 600

        with caplog.at_level(logging.WARNING):
            limit_batch_objects(targets, max_total=600)

        assert not any("RF-DETR batch object limiting" in record.message for record in caplog.records), (
            "No warning should be logged when no capping occurs"
        )

    def test_extreme_dense_image(self) -> None:
        """Test handling of Visdrone-like extreme dense image (1000+ objects)."""
        targets = self._create_targets([1000, 50, 50, 50])  # Visdrone-like scenario
        limited = limit_batch_objects(targets, max_total=600, max_per_image=300)

        total_after = sum(len(t["boxes"]) for t in limited)
        assert total_after <= 600, f"Total {total_after} exceeds limit 600"

        # Dense image should be capped at 300 first
        assert len(limited[0]["boxes"]) <= 300
        # Sparse images should be mostly preserved
        assert len(limited[1]["boxes"]) == 50
        assert len(limited[2]["boxes"]) == 50
        assert len(limited[3]["boxes"]) == 50


class TestRFDETRBatchLimitingHelpers:
    """Test class for batch limiting helper functions."""

    def test_get_largest_indices(self) -> None:
        """Test _get_largest_indices returns correct indices."""
        boxes = torch.tensor(
            [
                [0.5, 0.5, 0.1, 0.1],  # area = 0.01
                [0.5, 0.5, 0.3, 0.3],  # area = 0.09
                [0.5, 0.5, 0.2, 0.2],  # area = 0.04
            ]
        )
        indices = _get_largest_indices(boxes, k=2)
        assert indices.tolist() == [1, 2], "Should return indices of 2 largest boxes sorted"

    def test_subset_target(self) -> None:
        """Test _subset_target creates correct subset."""
        boxes = torch.rand(5, 4)
        target = {
            "boxes": boxes,
            "labels": torch.tensor([0, 1, 2, 3, 4]),
            "size": torch.tensor([512, 512]),  # Should not be subsetted
        }
        subset = _subset_target(target, torch.tensor([0, 2, 4]), original_length=5)
        assert len(subset["boxes"]) == 3
        assert len(subset["labels"]) == 3
        assert subset["labels"].tolist() == [0, 2, 4]
        assert subset["size"].tolist() == [512, 512]  # Unchanged

    def test_subset_target_with_masks(self) -> None:
        """Test _subset_target correctly filters masks for instance segmentation."""
        num_objects = 5
        boxes = torch.rand(num_objects, 4)
        masks = torch.rand(num_objects, 64, 64) > 0.5  # Binary masks (N, H, W)
        target = {
            "boxes": boxes,
            "labels": torch.tensor([0, 1, 2, 3, 4]),
            "masks": masks,
            "size": torch.tensor([512, 512]),
            "orig_size": torch.tensor([512, 512]),
        }
        indices = torch.tensor([1, 3])
        subset = _subset_target(target, indices, original_length=num_objects)

        assert len(subset["boxes"]) == 2
        assert len(subset["labels"]) == 2
        assert subset["labels"].tolist() == [1, 3]
        assert subset["masks"].shape == (2, 64, 64), "Masks should be filtered to (2, H, W)"
        assert subset["size"].tolist() == [512, 512]  # Unchanged

    def test_cap_per_image(self) -> None:
        """Test _cap_per_image caps images correctly."""
        targets = [
            {"boxes": torch.rand(500, 4), "labels": torch.randint(0, 10, (500,))},
            {"boxes": torch.rand(100, 4), "labels": torch.randint(0, 10, (100,))},
        ]
        capped = _cap_per_image(targets, max_per_image=200)
        assert len(capped[0]["boxes"]) == 200
        assert len(capped[1]["boxes"]) == 100  # Unchanged

    def test_reduce_dense_images(self) -> None:
        """Test _reduce_dense_images reduces only dense images."""
        targets = [
            {"boxes": torch.rand(100, 4), "labels": torch.randint(0, 10, (100,))},  # sparse
            {"boxes": torch.rand(200, 4), "labels": torch.randint(0, 10, (200,))},  # dense
        ]
        reduced = _reduce_dense_images(targets, max_total=250, sparse_threshold=150)
        # Sparse preserved (100), dense reduced (200 -> 150)
        assert len(reduced[0]["boxes"]) == 100
        assert len(reduced[1]["boxes"]) == 150

    def test_proportional_limit(self) -> None:
        """Test _proportional_limit reduces all images proportionally."""
        targets = [
            {"boxes": torch.rand(200, 4), "labels": torch.randint(0, 10, (200,))},
            {"boxes": torch.rand(200, 4), "labels": torch.randint(0, 10, (200,))},
        ]
        limited = _proportional_limit(targets, max_total=200)
        # Each should be reduced to ~100
        total = sum(len(t["boxes"]) for t in limited)
        assert total <= 200


class TestBatchLimitingWithMasks:
    """Test batch limiting with instance segmentation masks."""

    def _create_inst_seg_targets(
        self,
        object_counts: list[int],
        mask_size: tuple[int, int] = (64, 64),
    ) -> list[dict[str, torch.Tensor]]:
        """Create mock instance segmentation targets with masks.

        Args:
            object_counts: List of object counts per image.
            mask_size: Size of masks (H, W).

        Returns:
            List of target dicts with boxes, labels, and masks.
        """
        targets = []
        for count in object_counts:
            boxes = torch.rand(count, 4) * 0.5 + 0.1
            labels = torch.randint(0, 10, (count,))
            masks = torch.rand(count, *mask_size) > 0.5  # Binary masks
            targets.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "masks": masks,
                    "size": torch.tensor([512, 512]),
                    "orig_size": torch.tensor([512, 512]),
                }
            )
        return targets

    def test_masks_filtered_with_boxes(self) -> None:
        """Test that masks are filtered consistently with boxes."""
        targets = self._create_inst_seg_targets([500, 100])  # 600 total
        limited = limit_batch_objects(targets, max_total=400, max_per_image=300)

        for target in limited:
            num_boxes = len(target["boxes"])
            num_labels = len(target["labels"])
            num_masks = len(target["masks"])
            assert num_boxes == num_labels == num_masks, (
                f"Mismatch: boxes={num_boxes}, labels={num_labels}, masks={num_masks}"
            )

    def test_mask_shape_preserved(self) -> None:
        """Test that mask spatial dimensions (H, W) are preserved."""
        mask_h, mask_w = 64, 64
        targets = self._create_inst_seg_targets([500], mask_size=(mask_h, mask_w))
        limited = limit_batch_objects(targets, max_total=200, max_per_image=200)

        assert limited[0]["masks"].shape[1] == mask_h
        assert limited[0]["masks"].shape[2] == mask_w

    def test_inst_seg_per_image_cap(self) -> None:
        """Test per-image cap works with masks."""
        targets = self._create_inst_seg_targets([500, 50, 100])  # 650 total > 600
        limited = limit_batch_objects(targets, max_total=600, max_per_image=300)

        # Dense image capped to 300, bringing total to 450 <= 600
        assert len(limited[0]["boxes"]) == 300
        assert len(limited[0]["masks"]) == 300
        # Sparse images preserved
        assert len(limited[1]["boxes"]) == 50
        assert len(limited[2]["boxes"]) == 100

    def test_inst_seg_proportional_limit(self) -> None:
        """Test proportional limit with masks."""
        targets = self._create_inst_seg_targets([200, 200, 200])  # 600 total
        limited = limit_batch_objects(targets, max_total=300, max_per_image=300)

        total_after = sum(len(t["boxes"]) for t in limited)
        assert total_after <= 300

        # All masks should match their boxes
        for target in limited:
            assert len(target["masks"]) == len(target["boxes"])

    def test_empty_masks_handled(self) -> None:
        """Test handling of empty masks."""
        targets = [
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.long),
                "masks": torch.empty(0, 64, 64, dtype=torch.bool),
            },
            self._create_inst_seg_targets([100])[0],
        ]
        limited = limit_batch_objects(targets, max_total=600)

        assert len(limited[0]["boxes"]) == 0
        assert len(limited[0]["masks"]) == 0
        assert len(limited[1]["boxes"]) == 100

    def test_largest_objects_preserved_with_masks(self) -> None:
        """Test that largest objects (and their masks) are preserved."""
        # Create boxes with known areas
        boxes = torch.tensor(
            [
                [0.5, 0.5, 0.1, 0.1],  # area = 0.01 (smallest)
                [0.5, 0.5, 0.2, 0.2],  # area = 0.04
                [0.5, 0.5, 0.3, 0.3],  # area = 0.09
                [0.5, 0.5, 0.4, 0.4],  # area = 0.16 (largest)
            ]
        )
        # Create unique masks to identify which ones are kept
        masks = torch.zeros(4, 32, 32, dtype=torch.bool)
        for i in range(4):
            masks[i, i * 8 : (i + 1) * 8, :] = True  # Each mask has unique pattern

        targets = [
            {
                "boxes": boxes,
                "labels": torch.tensor([0, 1, 2, 3]),
                "masks": masks,
                "size": torch.tensor([512, 512]),
                "orig_size": torch.tensor([512, 512]),
            }
        ]

        limited = limit_batch_objects(targets, max_total=2, max_per_image=2)

        assert len(limited[0]["boxes"]) == 2
        assert len(limited[0]["masks"]) == 2
        # Labels 2 and 3 (largest) should be kept
        kept_labels = limited[0]["labels"].tolist()
        assert 3 in kept_labels
        assert 2 in kept_labels


class TestRFDETR:
    """Test class for RF-DETR detection model."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "rfdetr_nano",
            "rfdetr_small",
        ],
    )
    def test_init(self, model_name: str) -> None:
        """Test RF-DETR model initialization."""
        model = RFDETR(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=3,
            pretrained=False,
        )
        assert model.model_name == model_name
        assert model.num_classes == 3

    def test_create_model(self) -> None:
        """Test RF-DETR model creation."""
        model = RFDETR(
            model_name="rfdetr_medium",
            label_info=10,
            pretrained=False,
        )
        created_model = model._create_model()
        assert created_model is not None
        assert isinstance(created_model, torch.nn.Module)

        # Check if the model has the expected components
        assert hasattr(created_model, "lwdetr")
        assert hasattr(created_model, "criterion")
        assert hasattr(created_model, "postprocessor")

    def test_default_preprocessing_params(self) -> None:
        """Test default preprocessing parameters for different model variants."""
        model = RFDETR(
            model_name="rfdetr_medium",
            label_info=3,
            pretrained=False,
        )

        # Check that default params use 0-1 range normalization
        default_params = model._default_preprocessing_params
        assert "rfdetr_medium" in default_params
        assert default_params["rfdetr_medium"].input_size == (576, 576)
        # ImageNet mean in 0-1 range
        assert default_params["rfdetr_medium"].mean == (0.485, 0.456, 0.406)
        assert default_params["rfdetr_medium"].std == (0.229, 0.224, 0.225)

    def test_optimizer_configuration(self) -> None:
        """Test that optimizer configuration is properly set."""
        model = RFDETR(
            model_name="rfdetr_nano",
            label_info=5,
            pretrained=False,
        )

        # Test configure_optimizers method
        optimizers, schedulers = model.configure_optimizers()

        assert len(optimizers) == 1
        assert isinstance(optimizers[0], torch.optim.Optimizer)
        assert len(schedulers) > 0
        assert isinstance(schedulers, list)

        # Check that parameter groups are properly configured
        param_groups = optimizers[0].param_groups
        assert len(param_groups) > 0

    @pytest.mark.parametrize(
        ("model_name", "label_info"),
        [
            ("rfdetr_nano", 3),
        ],
    )
    def test_loss_computation(self, model_name: str, label_info: int, fxt_detection_batch) -> None:
        """Test RF-DETR loss computation in training mode."""
        input_sizes = {
            "rfdetr_nano": (384, 384),
            "rfdetr_small": (512, 512),
            "rfdetr_medium": (576, 576),
        }
        model = RFDETR(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=label_info,
            pretrained=False,
        )

        # Move model to CPU for unit tests
        model = model.cpu()
        input_size = input_sizes[model_name]
        # Get data batch
        fxt_detection_batch.images = torch.randn(2, 3, *input_size)

        # Set model to training mode
        model.train()

        # Forward pass should return loss dictionary
        output = model(fxt_detection_batch)

        # Check that output contains loss components
        assert isinstance(output, dict)

    @pytest.mark.parametrize(
        "model_name",
        [
            "rfdetr_nano",
        ],
    )
    def test_predict(self, model_name: str, fxt_detection_batch) -> None:
        """Test RF-DETR prediction in evaluation mode."""
        input_sizes = {
            "rfdetr_nano": (384, 384),
            "rfdetr_small": (512, 512),
            "rfdetr_medium": (576, 576),
        }
        input_size = input_sizes[model_name]

        model = RFDETR(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=3,
            pretrained=False,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Get data batch
        fxt_detection_batch.images = torch.randn(2, 3, *input_size)

        # Set model to evaluation mode
        model.eval()

        # Forward pass should return predictions
        output = model(fxt_detection_batch)

        # Check that output is PredictionBatch
        assert isinstance(output, PredictionBatch)
        assert output.batch_size == 2

    @pytest.mark.parametrize(
        "model_name",
        [
            "rfdetr_nano",
        ],
    )
    def test_export(self, model_name: str) -> None:
        """Test RF-DETR export functionality."""
        input_sizes = {
            "rfdetr_nano": (384, 384),
            "rfdetr_small": (512, 512),
            "rfdetr_medium": (576, 576),
        }
        input_size = input_sizes[model_name]

        model = RFDETR(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=3,
            pretrained=False,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Set model to evaluation mode
        model.eval()

        # Test export forward pass
        output = model.forward_for_tracing(torch.randn(1, 3, *input_size))
        assert len(output) == 3  # Should return boxes, labels, scores

    def test_multi_scale_training(self) -> None:
        """Test RF-DETR with multi-scale training enabled."""
        model = RFDETR(
            model_name="rfdetr_nano",
            label_info=3,
            multi_scale=True,
            pretrained=False,
        )

        # Move model to CPU for unit tests
        model = model.cpu()
        model = model.train()
        # Multi-scale should be enabled in the detector
        assert model.multi_scale is True
        assert isinstance(model.model.scales, list)
        assert len(model.model.scales) > 0

    def test_customize_inputs_handles_degenerate_empty_bboxes(self) -> None:
        """A (1, 0)-shaped empty bboxes tensor must not reach box_convert and crash."""
        model = RFDETR(
            model_name="rfdetr_nano",
            label_info=3,
            pretrained=False,
        )
        valid_bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            format="XYXY",
            canvas_size=(384, 384),
        )
        # validate_bboxes only inspects the first non-None entry, so the degenerate
        # shape must be on a later sample to reach the model's defensive guard.
        degenerate_bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.zeros((1, 0), dtype=torch.float32),
            format="XYXY",
            canvas_size=(384, 384),
        )
        entity = SampleBatch(
            images=torch.randn(2, 3, 384, 384),
            bboxes=[valid_bboxes, degenerate_bboxes],
            labels=[torch.zeros(1, dtype=torch.long), torch.zeros(0, dtype=torch.long)],
        )

        result = model._customize_inputs(entity)

        assert result["targets"][1]["boxes"].numel() == 0
