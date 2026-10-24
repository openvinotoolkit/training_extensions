# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``HFInstSegModel``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
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


def test_default_confidence_threshold_matches_export_default() -> None:
    model = HFInstSegModel(_tiny_config(), _label_info())

    assert model.default_confidence_threshold == pytest.approx(0.05)


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


def _tiny_eomt_dinov3_config() -> tf.EomtDinov3Config:
    return tf.EomtDinov3Config(
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=2,
        intermediate_size=64,
        num_blocks=2,  # small enough for an export-friendly capture
        num_queries=10,
        image_size=32,
        patch_size=16,
        num_register_tokens=0,
    )


def test_eomt_buffer_is_restored_after_tracing() -> None:
    """forward_for_tracing must not leave the EoMT mask-annealing buffer mutated."""
    model = HFInstSegModel(_tiny_eomt_dinov3_config(), _label_info())
    model.eval()
    original = model.hf_model.attn_mask_probs

    outputs = model.forward_for_tracing(torch.zeros(1, 3, 32, 32))

    assert outputs["masks"].shape[1] == 10
    assert model.hf_model.attn_mask_probs is original


def test_eomt_legacy_trace_export(tmp_path: Path) -> None:
    """EoMT must export via the legacy (dynamo=False) route.

    The EoMT eval branch ``self.training or self.attn_mask_probs[...] > 0``
    makes ``torch.export`` (dynamo) fail: buffers are lifted as graph inputs and
    the tensor-valued condition becomes a data-dependent guard. The legacy
    trace exporter evaluates the branch with real values and embeds the
    (annealed) constant decision, which is why the recipe pins
    ``onnx_dynamo: false`` — same as the Mask2Former family.
    """
    import tempfile

    model = HFInstSegModel(_tiny_eomt_dinov3_config(), _label_info())
    model.eval()

    import onnx

    original_model_forward = model.forward
    model.forward = model.forward_for_tracing  # type: ignore[method-assign]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eomt.onnx"
            torch.onnx.export(
                model,
                (torch.zeros(1, 3, 32, 32),),
                str(path),
                input_names=["images"],
                output_names=["boxes", "labels", "masks"],
                dynamo=False,
            )
            onnx_model = onnx.load(str(path))
            # all three outputs must be present in the traced graph
            assert {o.name for o in onnx_model.graph.output} == {"boxes", "labels", "masks"}
            assert path.stat().st_size > 0
    finally:
        model.forward = original_model_forward  # type: ignore[method-assign]
