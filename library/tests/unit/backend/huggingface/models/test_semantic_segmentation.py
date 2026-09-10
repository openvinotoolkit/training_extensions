# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``HFSemanticSegModel``."""

from __future__ import annotations

import math

import torch
import transformers as tf
from torchvision import tv_tensors

from getitune.backend.huggingface.models import HFSemanticSegModel
from getitune.data.entity.sample import SampleBatch
from getitune.types.label import LabelInfo, SegLabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


def _seg_label_info(ignore_index: int = 255) -> SegLabelInfo:
    return SegLabelInfo(
        label_names=["background", "road"],
        label_ids=["0", "1"],
        label_groups=[["background", "road"]],
        ignore_index=ignore_index,
    )


def _batch() -> SampleBatch:
    """Geti stores one (1, H, W) mask per sample; build_targets stacks them to (B, H, W)."""
    images = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    masks = [
        tv_tensors.Mask(torch.randint(0, 2, (1, 64, 64), dtype=torch.uint8)),
        tv_tensors.Mask(torch.randint(0, 2, (1, 64, 64), dtype=torch.uint8)),
    ]
    return SampleBatch(images=images, masks=masks)


def test_builds_from_config() -> None:
    model = HFSemanticSegModel(tf.SegformerConfig(), _seg_label_info())
    assert isinstance(model.hf_model, tf.SegformerForSemanticSegmentation)


def test_export_parameters() -> None:
    params = HFSemanticSegModel(tf.SegformerConfig(), _seg_label_info())._export_parameters
    assert params.model_type == "Segmentation"
    assert params.return_soft_prediction is True


def test_ignore_index_defaults_from_seg_label_info() -> None:
    model = HFSemanticSegModel(tf.SegformerConfig(), _seg_label_info(ignore_index=999))
    assert model.hf_model.config.semantic_loss_ignore_index == 999


def test_ignore_index_falls_back_to_255_for_plain_label_info() -> None:
    """A plain LabelInfo (no ignore_index attribute) falls back to 255."""
    model = HFSemanticSegModel(tf.SegformerConfig(), _label_info())
    assert model.hf_model.config.semantic_loss_ignore_index == 255


def test_explicit_ignore_index_wins_over_seg_label_info() -> None:
    model = HFSemanticSegModel(tf.SegformerConfig(), _seg_label_info(ignore_index=999), ignore_index=42)
    assert model.hf_model.config.semantic_loss_ignore_index == 42


class TestBuildTargets:
    def test_per_sample_masks_are_stacked_into_a_label_map(self) -> None:
        """G10: Geti's (1, H, W) per-sample masks become one (B, H, W) tensor."""
        model = HFSemanticSegModel(tf.SegformerConfig(), _seg_label_info())

        targets = model.build_targets(_batch())

        assert targets["labels"].shape == (2, 64, 64)
        assert targets["labels"].dtype == torch.long

    def test_forward_produces_a_finite_loss(self) -> None:
        model = HFSemanticSegModel(tf.SegformerConfig(), _seg_label_info())
        out = model.forward(_batch())
        assert math.isfinite(float(out.loss))  # type: ignore[attr-defined]
