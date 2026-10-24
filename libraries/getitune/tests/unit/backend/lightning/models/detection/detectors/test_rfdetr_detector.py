# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test of RFDETRDetector."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from getitune.backend.lightning.models.detection.detectors.rfdetr import RFDETRDetector


class DummyLWDETR(nn.Module):
    """Dummy LWDETR model for testing."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._export_mode = False
        # Dummy encoder with backbone for explain mode testing
        self.encoder = DummyEncoder()

    def forward(self, x, targets=None):
        """Forward pass returning pred_boxes, pred_logits, pred_masks."""
        batch = x.shape[0] if isinstance(x, torch.Tensor) else len(x.tensors) if hasattr(x, "tensors") else 2
        return {
            "pred_boxes": torch.rand(batch, 300, 4),
            "pred_logits": torch.rand(batch, 300, self.num_classes),
            "pred_masks": torch.rand(batch, 300, 16, 16),
        }

    def export(self):
        """Enable export mode."""
        self._export_mode = True

    def __call__(self, x, targets=None):
        """Make callable for export mode."""
        if self._export_mode:
            batch = x.shape[0] if isinstance(x, torch.Tensor) else 1
            # In export mode, return tuple format
            return (
                torch.rand(batch, 300, 4),  # pred_boxes
                torch.rand(batch, 300, self.num_classes),  # pred_logits
                torch.rand(batch, 300, 16, 16),  # pred_masks
            )
        return self.forward(x, targets)


class DummyEncoder(nn.Module):
    """Dummy encoder with backbone for explain mode testing."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DummyBackbone()


class DummyBackbone(nn.Module):
    """Dummy backbone for explain mode testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature map."""
        return self.conv(x)


class DummyCriterion(nn.Module):
    """Dummy criterion for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }

    def forward(self, outputs, targets):
        """Return dummy losses."""
        return {
            "loss_ce": torch.tensor(1.0),
            "loss_bbox": torch.tensor(0.5),
            "loss_giou": torch.tensor(0.3),
        }


class DummyPostprocessor(nn.Module):
    """Dummy postprocessor for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.num_select = 100

    def forward(self, outputs, target_sizes):
        """Return dummy detection results."""
        batch = outputs["pred_logits"].shape[0]
        return [
            {
                "scores": torch.rand(100),
                "boxes": torch.rand(100, 4) * 640,
                "labels": torch.randint(0, 10, (100,)),
                "masks": torch.randint(0, 2, (100, 1, 16, 16)),
            }
            for _ in range(batch)
        ]


class TestRFDETRDetector:
    """Test class for RFDETRDetector."""

    @pytest.fixture
    def rfdetr_detector(self) -> RFDETRDetector:
        """Create RFDETRDetector instance for testing."""
        rfdetr_args = SimpleNamespace(
            resolution=560,
            expanded_scales=[0.5, 1.0, 1.5],
            patch_size=16,
            num_windows=4,
        )
        return RFDETRDetector(
            lwdetr_model=DummyLWDETR(num_classes=10),
            criterion=DummyCriterion(),
            postprocessor=DummyPostprocessor(),
            rfdetr_args=rfdetr_args,  # pyrefly: ignore[bad-argument-type]
            input_size=560,
            multi_scale=False,
        )

    @pytest.fixture
    def targets(self) -> list[dict]:
        """Create dummy targets for testing."""
        return [
            {
                "boxes": torch.tensor([[0.2739, 0.2848, 0.3239, 0.3348], [0.1652, 0.1109, 0.2152, 0.1609]]),
                "labels": torch.tensor([2, 2]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [0.6761, 0.8174, 0.7261, 0.8674],
                        [0.1652, 0.1109, 0.2152, 0.1609],
                        [0.2848, 0.9370, 0.3348, 0.9870],
                    ],
                ),
                "labels": torch.tensor([8, 2, 7]),
            },
        ]

    @pytest.fixture
    def images(self) -> torch.Tensor:
        """Create dummy images for testing."""
        return torch.randn(2, 3, 560, 560)

    def test_init(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test RFDETRDetector initialization."""
        assert rfdetr_detector.input_size == 560
        # multi_scale is represented via the scales list
        assert isinstance(rfdetr_detector.scales, list)
        assert rfdetr_detector.lwdetr is not None
        assert rfdetr_detector.criterion is not None
        assert rfdetr_detector.postprocessor is not None

    def test_init_with_multi_scale(self) -> None:
        """Test RFDETRDetector initialization with multi-scale training."""
        rfdetr_args = SimpleNamespace(
            resolution=560,
            expanded_scales=[0.5, 1.0, 1.5],
            patch_size=16,
            num_windows=4,
        )
        detector = RFDETRDetector(
            lwdetr_model=DummyLWDETR(),
            criterion=DummyCriterion(),
            postprocessor=DummyPostprocessor(),
            rfdetr_args=rfdetr_args,  # pyrefly: ignore[bad-argument-type]
            input_size=560,
            multi_scale=True,
        )
        assert len(detector.scales) > 0

    def test_generate_scales(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test scale generation for multi-scale training."""
        # Ensure that scales is a list even when multi_scale is False
        assert isinstance(rfdetr_detector.scales, list)

    def test_postprocess(self, rfdetr_detector: RFDETRDetector) -> None:
        """Test postprocess method."""
        outputs = {
            "pred_logits": torch.randn(2, 300, 10),
            "pred_boxes": torch.randn(2, 300, 4),
            "pred_masks": torch.randn(2, 300, 16, 16),
        }
        original_sizes = [(640, 640), (640, 640)]
        scores, boxes, labels, masks = rfdetr_detector.postprocess(outputs, original_sizes)

        assert isinstance(scores, list)
        assert isinstance(boxes, list)
        assert isinstance(labels, list)
        assert isinstance(masks, list)
        assert len(scores) == 2
        assert len(boxes) == 2
        assert len(labels) == 2
        assert len(masks) == 2

    def test_export(self, rfdetr_detector: RFDETRDetector, images: torch.Tensor) -> None:
        """Test export method."""
        rfdetr_detector.eval()
        # Ensure underlying model is set to export mode if required
        if hasattr(rfdetr_detector.lwdetr, "export"):
            rfdetr_detector.lwdetr.export()  # pyrefly: ignore[not-callable]

        result = rfdetr_detector.export(images)

        # Check that export returned expected tuple
        assert isinstance(result, tuple)
        assert len(result) == 4
        boxes, labels, scores, masks = result
        assert boxes.shape[0] == 2
        assert labels.shape[0] == 2
        assert scores.shape[0] == 2
        assert masks.shape[0] == 2
        # Boxes must be xyxy, not cxcywh
        # In valid xyxy format: x1 <= x2 and y1 <= y2 for all boxes
        assert (boxes[:, :, 0] <= boxes[:, :, 2]).all()
        assert (boxes[:, :, 1] <= boxes[:, :, 3]).all()
        # Default boxes are 4-column xyxy.
        assert boxes.shape[-1] == 4

    def test_export_merge_scores_with_masks(
        self,
        rfdetr_detector: RFDETRDetector,
        images: torch.Tensor,
    ) -> None:
        """``merge_scores=True`` must concat scores into ``boxes`` and drop scores tensor.

        This is the contract required by the OpenVINO ``MaskRCNN`` model_api
        wrapper used by the instance-segmentation export path, which reads
        ``outputs["boxes"][:, 4]`` for the confidence score.
        """
        rfdetr_detector.eval()
        if hasattr(rfdetr_detector.lwdetr, "export"):
            rfdetr_detector.lwdetr.export()  # pyrefly: ignore[not-callable]

        result = rfdetr_detector.export(images, merge_scores=True)

        assert isinstance(result, tuple)
        assert len(result) == 3
        boxes_with_scores, labels, masks = result
        # Boxes must now carry the score in the 5th column.
        assert boxes_with_scores.ndim == 3
        assert boxes_with_scores.shape[0] == 2
        assert boxes_with_scores.shape[-1] == 5
        # Scores (last column) must be in [0, 1] (sigmoid of pred_logits).
        last_col = boxes_with_scores[..., 4]
        assert torch.all(last_col >= 0.0)
        assert torch.all(last_col <= 1.0)
        # xyxy validity on the first 4 columns.
        assert (boxes_with_scores[..., 0] <= boxes_with_scores[..., 2]).all()
        assert (boxes_with_scores[..., 1] <= boxes_with_scores[..., 3]).all()
        assert labels.shape[:2] == boxes_with_scores.shape[:2]
        assert masks.shape[:2] == boxes_with_scores.shape[:2]
