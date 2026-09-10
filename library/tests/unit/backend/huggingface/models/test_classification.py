# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``HFMulticlassClsModel`` and ``HFMultilabelClsModel``."""

from __future__ import annotations

import math

import torch
import transformers as tf

from getitune.backend.huggingface.models import HFMulticlassClsModel, HFMultilabelClsModel
from getitune.data.entity.sample import SampleBatch
from getitune.types.label import LabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(
        label_names=["cat", "dog", "bird"], label_ids=["0", "1", "2"], label_groups=[["cat", "dog", "bird"]]
    )


def _tiny_config() -> tf.ViTConfig:
    return tf.ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)


def _multiclass_batch() -> SampleBatch:
    images = torch.rand(2, 3, 224, 224, dtype=torch.float32)
    labels = [torch.tensor(0), torch.tensor(2)]
    return SampleBatch(images=images, labels=labels)


def _multilabel_batch() -> SampleBatch:
    images = torch.rand(2, 3, 224, 224, dtype=torch.float32)
    labels = [torch.tensor([1, 0, 1], dtype=torch.long), torch.tensor([0, 1, 0], dtype=torch.long)]
    return SampleBatch(images=images, labels=labels)


def test_multiclass_builds_from_config() -> None:
    model = HFMulticlassClsModel(_tiny_config(), _label_info())
    assert isinstance(model.hf_model, tf.ViTForImageClassification)
    assert model.hf_model.config.problem_type in (None, "single_label_classification")


def test_multiclass_export_parameters() -> None:
    params = HFMulticlassClsModel(_tiny_config(), _label_info())._export_parameters
    assert params.model_type == "Classification"
    assert params.multilabel is False


def test_multilabel_sets_problem_type() -> None:
    model = HFMultilabelClsModel(_tiny_config(), _label_info())
    assert isinstance(model.hf_model, tf.ViTForImageClassification)
    assert model.hf_model.config.problem_type == "multi_label_classification"


def test_multilabel_export_parameters() -> None:
    params = HFMultilabelClsModel(_tiny_config(), _label_info())._export_parameters
    assert params.model_type == "Classification"
    assert params.multilabel is True


class TestBuildTargets:
    def test_multiclass_stacks_scalar_labels(self) -> None:
        model = HFMulticlassClsModel(_tiny_config(), _label_info())

        targets = model.build_targets(_multiclass_batch())

        assert targets["labels"].shape == (2,)
        assert targets["labels"].dtype == torch.long
        torch.testing.assert_close(targets["labels"], torch.tensor([0, 2]))

    def test_multiclass_forward_produces_a_finite_loss(self) -> None:
        model = HFMulticlassClsModel(_tiny_config(), _label_info())
        out = model.forward(_multiclass_batch())
        assert math.isfinite(float(out.loss))  # type: ignore[attr-defined]

    def test_multilabel_stacks_multi_hot_vectors_as_float(self) -> None:
        model = HFMultilabelClsModel(_tiny_config(), _label_info())

        targets = model.build_targets(_multilabel_batch())

        assert targets["labels"].shape == (2, 3)
        assert targets["labels"].dtype == torch.float32
        torch.testing.assert_close(targets["labels"], torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))

    def test_multilabel_forward_produces_a_finite_loss(self) -> None:
        model = HFMultilabelClsModel(_tiny_config(), _label_info())
        out = model.forward(_multilabel_batch())
        assert math.isfinite(float(out.loss))  # type: ignore[attr-defined]
