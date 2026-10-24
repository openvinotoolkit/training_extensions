# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RF-DETR instance segmentation model."""

from __future__ import annotations

import torch

from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.common.rfdetr_mixin import RFDETRMixin
from getitune.backend.lightning.models.instance_segmentation.rfdetr_inst import RFDETRInst
from getitune.data.entity import PredictionBatch


class TestRFDETRInst:
    """Test class for RF-DETR instance segmentation model."""

    def test_init(self) -> None:
        """Test RF-DETR instance segmentation model initialization."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            pretrained=False,
        )
        assert model.model_name == "rfdetr_seg_n"
        assert model.num_classes == 3

    def test_create_model(self) -> None:
        """Test RF-DETR instance segmentation model creation."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
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
        """Test default preprocessing parameters for RF-DETR instance segmentation."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            pretrained=False,
        )

        # Check that default params use 0-1 range normalization
        default_params = model._default_preprocessing_params
        assert "rfdetr_seg_n" in default_params
        assert default_params["rfdetr_seg_n"].input_size == (312, 312)
        # ImageNet mean in 0-1 range
        assert default_params["rfdetr_seg_n"].mean == (0.485, 0.456, 0.406)
        assert default_params["rfdetr_seg_n"].std == (0.229, 0.224, 0.225)

    def test_optimizer_configuration(self) -> None:
        """Test that optimizer configuration is properly set."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
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

    def test_loss_computation(self, fxt_instance_seg_batch) -> None:
        """Test RF-DETR instance segmentation loss computation in training mode."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            data_input_params=DataInputParams((312, 312), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Adjust batch images to match input size
        fxt_instance_seg_batch.images = [
            torch.randn(3, 312, 312),
            torch.randn(3, 312, 312),
        ]

        # Set model to training mode
        model.train()

        # Forward pass should return loss dictionary
        output = model(fxt_instance_seg_batch)

        # Check that output contains loss components
        assert isinstance(output, dict)
        # RF-DETR segmentation should have mask losses in addition to detection losses
        assert any("loss" in key for key in output)

        # Check that loss values are not None and are valid tensors
        for key, value in output.items():
            if "loss" in key:
                assert value is not None
                assert isinstance(value, torch.Tensor)
                assert not torch.isnan(value)
                assert not torch.isinf(value)

    def test_predict(self, fxt_instance_seg_batch) -> None:
        """Test RF-DETR instance segmentation prediction in evaluation mode."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            data_input_params=DataInputParams((312, 312), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Adjust batch images to match input size
        fxt_instance_seg_batch.images = [
            torch.randn(3, 312, 312),
            torch.randn(3, 312, 312),
        ]

        # Set model to evaluation mode
        model.eval()

        # Forward pass should return predictions
        output = model(fxt_instance_seg_batch)

        # Check that output is PredictionBatch with masks
        assert isinstance(output, PredictionBatch)
        assert output.batch_size == 2
        assert output.masks is not None
        assert len(output.masks) == 2

    def test_export(self) -> None:
        """Test RF-DETR instance segmentation export functionality."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            pretrained=False,
        )

        # Move model to CPU for unit tests
        model = model.cpu()

        # Set model to evaluation mode
        model.eval()

        # Test export forward pass
        output = model.forward_for_tracing(torch.randn(1, 3, 312, 312))
        # Should return dict with boxes, labels, masks where ``boxes`` has scores
        # concatenated as the 5th column to match the OpenVINO ``MaskRCNN``
        # model_api wrapper's expectation of ``boxes[:, 4]`` being the score.
        assert isinstance(output, dict)
        assert len(output) == 3
        boxes = output["boxes"]
        labels = output["labels"]
        masks = output["masks"]
        assert boxes.ndim == 3
        assert boxes.shape[-1] == 5  # x1, y1, x2, y2, score
        assert labels.shape[:2] == boxes.shape[:2]
        assert masks.shape[:2] == boxes.shape[:2]

    def test_exporter_output_names(self) -> None:
        """Exporter must publish ``boxes``/``labels``/``masks`` (no standalone ``scores``)."""
        model = RFDETRInst(model_name="rfdetr_seg_n", label_info=3, pretrained=False)
        exporter = model._exporter
        assert isinstance(exporter, LightningModelExporter)
        assert exporter.output_names == ["boxes", "labels", "masks"]
        onnx_cfg = exporter.onnx_export_configuration
        assert onnx_cfg["output_names"] == ["boxes", "labels", "masks"]

    def test_predict_after_export_restore(self, fxt_instance_seg_batch) -> None:
        """Inference must work after export() + _restore_forward_methods().

        Regression test for a bug where ``TransformerDecoder._export`` was not
        reset by ``_restore_forward_methods`` because the decoder has no
        ``_forward_origin``.  The lingering flag made the decoder return a 3-D
        tensor instead of 4-D, crashing the segmentation head's einsum.
        """
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            data_input_params=DataInputParams((312, 312), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        model = model.cpu()
        model.eval()

        fxt_instance_seg_batch.images = [
            torch.randn(3, 312, 312),
            torch.randn(3, 312, 312),
        ]

        # Simulate what RFDETRMixin.export() does: export then restore.
        lwdetr = model.model.lwdetr  # pyrefly: ignore[missing-attribute]
        lwdetr.export()  # pyrefly: ignore[missing-attribute]
        RFDETRMixin._restore_forward_methods(lwdetr)  # pyrefly: ignore[bad-argument-type]

        # Verify the decoder flag was cleared.
        assert not lwdetr.transformer.decoder._export  # pyrefly: ignore[missing-attribute]

        # Inference must succeed after the round-trip.
        output = model(fxt_instance_seg_batch)
        assert isinstance(output, PredictionBatch)
        assert output.masks is not None

    def test_customize_inputs(self, fxt_instance_seg_batch) -> None:
        """Test input customization for RF-DETR format."""
        model = RFDETRInst(
            model_name="rfdetr_seg_n",
            label_info=3,
            pretrained=False,
        )

        customized = model._customize_inputs(fxt_instance_seg_batch)

        # Check that customized inputs have the expected format
        assert "images" in customized
        assert "targets" in customized
        assert isinstance(customized["targets"], list)
        assert len(customized["targets"]) == fxt_instance_seg_batch.batch_size

        # Check target structure
        for target in customized["targets"]:
            assert "boxes" in target
            assert "labels" in target
            assert "masks" in target
            assert "size" in target
            assert "orig_size" in target

    def test_customize_inputs_aligns_mismatched_annotation_counts(self) -> None:
        """Mismatched per-image boxes/labels/masks are aligned to a common count.

        Regression test for a crash inside the RF-DETR Hungarian matcher
        (``RuntimeError: The size of tensor a (N) must match the size of tensor
        b (M) ...``) that occurred when a geometric/tiling transform dropped one
        annotation type (e.g. boxes) without dropping the paired masks. The
        criterion concatenates boxes/labels/masks per target and requires their
        counts to match, so ``_customize_inputs`` must align them.
        """
        from torchvision import tv_tensors

        from getitune.data.entity.base import ImageInfo
        from getitune.data.entity.sample import SampleBatch

        model = RFDETRInst(model_name="rfdetr_seg_n", label_info=3)

        # Image 0: 1 box / 1 label but 3 masks (masks not dropped with boxes).
        # Image 1: 2 masks / 2 labels but only 1 box.
        batch = SampleBatch(
            images=torch.stack([torch.randn(3, 320, 320), torch.randn(3, 320, 320)]),
            bboxes=[
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(320, 320),
                ),
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    torch.tensor([[20, 20, 80, 80]], dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(320, 320),
                ),
            ],
            labels=[
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1, 2], dtype=torch.long),
            ],
            masks=[
                tv_tensors.Mask(torch.zeros((3, 320, 320), dtype=torch.uint8)),
                tv_tensors.Mask(torch.zeros((2, 320, 320), dtype=torch.uint8)),
            ],
            imgs_info=[
                ImageInfo(  # pyrefly: ignore[no-matching-overload]
                    img_idx=0, img_shape=(320, 320), ori_shape=(320, 320)
                ),
                ImageInfo(  # pyrefly: ignore[no-matching-overload]
                    img_idx=1, img_shape=(320, 320), ori_shape=(320, 320)
                ),
            ],
        )

        customized = model._customize_inputs(batch)
        targets = customized["targets"]
        assert len(targets) == 2

        # Every target must have matching boxes/labels/masks counts so the
        # criterion's per-target concatenation does not raise.
        for target in targets:
            n_boxes = target["boxes"].shape[0]
            assert target["labels"].shape[0] == n_boxes
            assert target["masks"].shape[0] == n_boxes

        # Counts are aligned to the per-image minimum (1 for both images here).
        assert targets[0]["boxes"].shape[0] == 1
        assert targets[1]["boxes"].shape[0] == 1

    def test_customize_inputs_rebuilds_boxes_from_masks_on_mismatch(self) -> None:
        """On a boxes/masks mismatch, RF-DETR targets keep box↔mask correspondence.

        The matcher pairs per-target box and mask costs, so after alignment
        ``box[i]`` must be the tight box of ``mask[i]``. Trimming boxes to the
        first ``n`` would pair a box with the wrong mask when a middle instance
        is dropped from only one field, so the boxes are recomputed from masks.
        """
        from torchvision import tv_tensors

        from getitune.data.entity.base import ImageInfo
        from getitune.data.entity.sample import SampleBatch

        model = RFDETRInst(model_name="rfdetr_seg_n", label_info=3)
        size = 320
        # Two masks with distinct rectangles, but only one (unrelated) stored box.
        mask_rects = [(10, 20, 60, 90), (100, 40, 180, 200)]
        mask_data = torch.zeros((2, size, size), dtype=torch.uint8)
        for idx, (x1, y1, x2, y2) in enumerate(mask_rects):
            mask_data[idx, y1:y2, x1:x2] = 1
        batch = SampleBatch(
            images=torch.randn(1, 3, size, size),
            bboxes=[
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    torch.tensor([[0, 0, 5, 5]], dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(size, size),
                )
            ],
            labels=[torch.tensor([0], dtype=torch.long)],
            masks=[tv_tensors.Mask(mask_data)],
            imgs_info=[
                ImageInfo(  # pyrefly: ignore[no-matching-overload]
                    img_idx=0, img_shape=(size, size), ori_shape=(size, size)
                )
            ],
        )

        target = model._customize_inputs(batch)["targets"][0]
        # Aligned to the common minimum (1 box vs 2 masks -> 1).
        assert target["boxes"].shape[0] == 1
        assert target["masks"].shape[0] == 1
        # The single kept box must be the (normalized cxcywh) tight box of mask 0,
        # NOT the unrelated stored [0, 0, 5, 5] box.
        x1, y1, x2, y2 = mask_rects[0]
        exp_cxcywh = torch.tensor(
            [
                (x1 + (x2 - 1)) / 2 / size,
                (y1 + (y2 - 1)) / 2 / size,
                (x2 - 1 - x1) / size,
                (y2 - 1 - y1) / size,
            ]
        )
        torch.testing.assert_close(target["boxes"][0].cpu(), exp_cxcywh, rtol=1e-4, atol=1e-4)
