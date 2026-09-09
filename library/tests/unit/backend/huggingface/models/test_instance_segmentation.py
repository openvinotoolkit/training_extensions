# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``HFInstSegModel``."""

from __future__ import annotations

import math

import torch
import transformers as tf
from torchvision import tv_tensors

from getitune.backend.huggingface.models import HFInstSegModel
from getitune.data.entity.sample import SampleBatch
from getitune.types.label import LabelInfo

_LABEL_NAMES = ["cat", "dog", "bird"]


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=list(_LABEL_NAMES), label_ids=["0", "1", "2"], label_groups=[list(_LABEL_NAMES)])


def _tiny_config() -> tf.Mask2FormerConfig:
    return tf.Mask2FormerConfig(num_queries=10)


def _batch() -> SampleBatch:
    """One image with two instance masks, one image with none (G13)."""
    images = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    masks = [
        tv_tensors.Mask(torch.randint(0, 2, (2, 32, 32), dtype=torch.uint8)),
        tv_tensors.Mask(torch.zeros((0, 32, 32), dtype=torch.uint8)),
    ]
    labels = [torch.tensor([0, 2], dtype=torch.long), torch.zeros(0, dtype=torch.long)]
    return SampleBatch(images=images, masks=masks, labels=labels)


def test_builds_from_config() -> None:
    model = HFInstSegModel(_tiny_config(), _label_info())
    assert isinstance(model.hf_model, tf.Mask2FormerForUniversalSegmentation)


def test_export_parameters_shift_labels() -> None:
    """ModelAPI's DETRInstSeg wrapper expects a leading placeholder label (G21)."""
    model = HFInstSegModel(_tiny_config(), _label_info())

    params = model._export_parameters

    assert params.model_type == "DETRInstSeg"
    assert params.label_info.label_names[0] == "getitune_empty_lbl"
    assert params.label_info.label_names[1:] == _LABEL_NAMES
    # the shift must not leak back into the model's own label_info
    assert model.label_info.label_names == _LABEL_NAMES


class TestBuildTargets:
    def test_masks_become_float_and_labels_stay_long(self) -> None:
        model = HFInstSegModel(_tiny_config(), _label_info())

        targets = model.build_targets(_batch())

        assert targets["mask_labels"][0].dtype == torch.float32
        assert targets["mask_labels"][0].shape == (2, 32, 32)
        assert targets["class_labels"][0].dtype == torch.long

    def test_empty_image_yields_zero_count_masks_and_labels(self) -> None:
        """G13/G9: an image with no instances, and masks converted from uint8."""
        model = HFInstSegModel(_tiny_config(), _label_info())

        targets = model.build_targets(_batch())

        assert targets["mask_labels"][1].shape == (0, 32, 32)
        assert targets["class_labels"][1].shape == (0,)

    def test_forward_produces_a_finite_loss(self) -> None:
        model = HFInstSegModel(_tiny_config(), _label_info())
        out = model.forward(_batch())
        assert math.isfinite(float(out.loss))  # type: ignore[attr-defined]
