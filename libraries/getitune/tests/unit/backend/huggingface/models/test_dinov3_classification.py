# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DINOv3 classification wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers as tf

from getitune.backend.huggingface.models import HFDinov3MulticlassClsModel, HFDinov3MultilabelClsModel
from getitune.data.entity.sample import SampleBatch
from getitune.types.label import LabelInfo


def _labels() -> LabelInfo:
    return LabelInfo(
        label_names=["cat", "dog", "bird"], label_ids=["0", "1", "2"], label_groups=[["cat", "dog", "bird"]]
    )


def _config() -> tf.DINOv3ViTConfig:
    return tf.DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
        id2label={0: "cat", 1: "dog", 2: "bird"},
        label2id={"cat": 0, "dog": 1, "bird": 2},
    )


def test_multiclass_forward() -> None:
    model = HFDinov3MulticlassClsModel(_config(), _labels(), data_input_params={"input_size": (32, 32)})
    batch = SampleBatch(images=torch.rand(2, 3, 32, 32), labels=[torch.tensor(0), torch.tensor(1)])
    outputs = model(batch)
    assert outputs.logits.shape == (2, 3)
    assert outputs.loss is not None


def test_multilabel_forward() -> None:
    model = HFDinov3MultilabelClsModel(_config(), _labels(), data_input_params={"input_size": (32, 32)})
    batch = SampleBatch(
        images=torch.rand(2, 3, 32, 32),
        labels=[torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])],
    )
    outputs = model(batch)
    assert outputs.logits.shape == (2, 3)
    assert outputs.loss is not None


def test_export_contract() -> None:
    model = HFDinov3MulticlassClsModel(_config(), _labels())
    assert model._onnx_output_names == ["logits"]
    assert model._export_parameters.model_type == "Classification"


def test_backbone_trainable_by_default() -> None:
    model = HFDinov3MulticlassClsModel(_config(), _labels(), data_input_params={"input_size": (32, 32)})
    assert all(p.requires_grad for p in model.hf_model.backbone.parameters())
    assert all(p.requires_grad for p in model.hf_model.classifier.parameters())


def test_backbone_frozen_when_overridden() -> None:
    model = HFDinov3MulticlassClsModel(
        _config(),
        _labels(),
        data_input_params={"input_size": (32, 32)},
        extra_overrides={"freeze_backbone": True},
    )
    assert all(not p.requires_grad for p in model.hf_model.backbone.parameters())


def test_local_raw_snapshot_loads_backbone_through_transformers(tmp_path: Path) -> None:
    raw_snapshot = tmp_path / "raw"
    tf.AutoModel.from_config(_config()).save_pretrained(raw_snapshot)

    model = HFDinov3MulticlassClsModel(
        "timm/dinov3-architecture",
        _labels(),
        pretrained_weights=raw_snapshot,
        data_input_params={"input_size": (32, 32)},
    )

    assert model.checkpoint == "timm/dinov3-architecture"
    assert all(parameter.requires_grad for parameter in model.hf_model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.hf_model.classifier.parameters())


def test_getitune_checkpoint_marker_and_round_trip(tmp_path: Path) -> None:
    original = HFDinov3MulticlassClsModel(_config(), _labels(), data_input_params={"input_size": (32, 32)})
    original.hf_model.save_pretrained(tmp_path)

    saved_config = json.loads((tmp_path / "config.json").read_text())
    restored = HFDinov3MulticlassClsModel(
        "timm/dinov3-architecture",
        _labels(),
        pretrained_weights=tmp_path,
        data_input_params={"input_size": (32, 32)},
    )

    assert saved_config["getitune_model_type"] == "dinov3_image_classifier"
    for name, parameter in original.hf_model.state_dict().items():
        assert torch.equal(parameter, restored.hf_model.state_dict()[name])


def test_getitune_checkpoint_replaces_head_when_labels_change(tmp_path: Path) -> None:
    original = HFDinov3MulticlassClsModel(_config(), _labels(), input_size=(32, 32))
    original.hf_model.save_pretrained(tmp_path)
    new_labels = LabelInfo(
        label_names=["cat", "dog", "bird", "fish"],
        label_ids=["0", "1", "2", "3"],
        label_groups=[["cat", "dog", "bird", "fish"]],
    )

    restored = HFDinov3MulticlassClsModel(
        "timm/dinov3-architecture", new_labels, pretrained_weights=tmp_path, input_size=(32, 32)
    )

    assert restored.hf_model.classifier.out_features == 4
    assert restored.hf_model.config.id2label == dict(enumerate(new_labels.label_names))
