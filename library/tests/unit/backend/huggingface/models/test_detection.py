# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``HFDetectionModel``."""

from __future__ import annotations

import math

import torch
import transformers as tf
from torchvision import tv_tensors

from getitune.backend.huggingface.models import HFDetectionModel
from getitune.data.entity.sample import SampleBatch
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType


def _label_info() -> LabelInfo:
    return LabelInfo(
        label_names=["cat", "dog", "bird"], label_ids=["0", "1", "2"], label_groups=[["cat", "dog", "bird"]]
    )


def _tiny_config() -> tf.RTDetrV2Config:
    return tf.RTDetrV2Config(num_queries=10, decoder_layers=2)


def _batch() -> SampleBatch:
    """One image with two boxes, one image with none (G13)."""
    images = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    bboxes = [
        tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.tensor([[10.0, 10.0, 30.0, 30.0], [5.0, 5.0, 20.0, 20.0]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(64, 64),
        ),
        tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.zeros((0, 4)), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(64, 64)
        ),
    ]
    labels = [torch.tensor([0, 2], dtype=torch.long), torch.zeros(0, dtype=torch.long)]
    return SampleBatch(images=images, bboxes=bboxes, labels=labels)


def test_builds_from_config() -> None:
    model = HFDetectionModel(_tiny_config(), _label_info())
    assert isinstance(model.hf_model, tf.RTDetrV2ForObjectDetection)
    assert model.hf_model.config.id2label == {0: "cat", 1: "dog", 2: "bird"}
    assert model.hf_model.config.label2id == {"cat": 0, "dog": 1, "bird": 2}
    assert model.task == TaskType.DETECTION


def test_export_parameters() -> None:
    params = HFDetectionModel(_tiny_config(), _label_info())._export_parameters
    assert params.model_type == "ssd"
    assert params.task_type == "detection"


class TestBuildTargets:
    def test_boxes_are_normalized_cxcywh(self) -> None:
        model = HFDetectionModel(_tiny_config(), _label_info())

        targets = model.build_targets(_batch())

        assert targets["pixel_values"].shape == (2, 3, 64, 64)
        first = targets["labels"][0]
        torch.testing.assert_close(first["class_labels"], torch.tensor([0, 2]))
        torch.testing.assert_close(
            first["boxes"],
            torch.tensor([[20 / 64, 20 / 64, 20 / 64, 20 / 64], [12.5 / 64, 12.5 / 64, 15 / 64, 15 / 64]]),
        )

    def test_empty_image_yields_zero_count_boxes_and_labels(self) -> None:
        """G13: an image with no annotations must not break the conversion."""
        model = HFDetectionModel(_tiny_config(), _label_info())

        targets = model.build_targets(_batch())

        second = targets["labels"][1]
        assert second["boxes"].shape == (0, 4)
        assert second["class_labels"].shape == (0,)

    def test_forward_produces_a_finite_loss(self) -> None:
        model = HFDetectionModel(_tiny_config(), _label_info())
        out = model.forward(_batch())
        assert math.isfinite(float(out.loss))  # type: ignore[attr-defined]
