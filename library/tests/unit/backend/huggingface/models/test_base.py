# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared ``HFModel`` base behavior.

Classification is used as a concrete stand-in to exercise the generic
behavior implemented in
``getitune.backend.huggingface.models.base.HFModel`` (label dispatch,
checkpoint loading, ``data_input_params``, ``_export_parameters``, and the
shared ``forward`` -> ``build_targets`` delegation). Detection, instance
segmentation, semantic segmentation, and classification each get their own
file for ``build_targets`` behavior specific to that task.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import transformers as tf

from getitune.backend.huggingface.models import HFModel, HFMulticlassClsModel
from getitune.data.entity.sample import SampleBatch
from getitune.types.label import LabelInfo, SegLabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(
        label_names=["cat", "dog", "bird"], label_ids=["0", "1", "2"], label_groups=[["cat", "dog", "bird"]]
    )


def _tiny_vit_config() -> tf.ViTConfig:
    return tf.ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)


def _batch() -> SampleBatch:
    images = torch.rand(2, 3, 224, 224, dtype=torch.float32)
    labels = [torch.tensor(0), torch.tensor(2)]
    return SampleBatch(images=images, labels=labels)


def test_hf_model_is_a_registered_submodule() -> None:
    """hf_model must show up in .parameters(), or gradients won't flow."""
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info())
    assert any(p is model.hf_model.parameters().__next__() for p in model.parameters())


def test_imgsz_derives_from_data_input_params() -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info(), data_input_params={"input_size": (224, 224)})
    assert model.imgsz == 224
    assert model.data_input_params.input_size == (224, 224)


def test_best_checkpoint_starts_as_none() -> None:
    assert HFMulticlassClsModel(_tiny_vit_config(), _label_info()).best_checkpoint is None


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info())
    model.hf_model.save_pretrained(tmp_path)

    model.load_checkpoint(tmp_path)

    assert model.best_checkpoint == tmp_path
    assert isinstance(model.hf_model, tf.ViTForImageClassification)


def test_set_intensity_config_stores_value() -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info())
    sentinel = object()
    model.set_intensity_config(sentinel)  # type: ignore[arg-type]
    assert model._intensity_config is sentinel


def test_ensure_predict_ready_switches_to_eval() -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info())
    model.train()
    model.ensure_predict_ready()
    assert model.training is False


def test_exporter_builds_a_configured_hf_model_exporter() -> None:
    from getitune.backend.huggingface.exporter.hf_exporter import HFModelExporter

    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info())
    exporter = model._exporter
    assert isinstance(exporter, HFModelExporter)
    assert exporter.task_level_export_parameters.model_type == "Classification"


def test_resize_mode_is_passed_to_exporter() -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info(), resize_mode="fit_to_window")
    assert model._exporter.resize_mode == "fit_to_window"


def test_forward_runs_end_to_end_through_build_targets() -> None:
    """forward() is shared: build_targets is implemented, so this must now work."""
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info())
    out = model.forward(_batch())
    assert math.isfinite(float(out.loss))  # type: ignore[attr-defined]


def test_forward_for_tracing_returns_raw_logits() -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info(), data_input_params={"input_size": (224, 224)})
    images = torch.rand(2, 3, 224, 224)
    logits = model.forward_for_tracing(images)
    assert logits.shape == (2, 3)


def test_export_writes_an_onnx_file(tmp_path: Path) -> None:
    from getitune.types.export import ExportFormat

    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info(), data_input_params={"input_size": (224, 224)})
    path = model.export(tmp_path, "exported_model", ExportFormat.ONNX)
    assert path == tmp_path / "exported_model.onnx"
    assert path.exists()


def test_export_restores_training_mode_and_device(tmp_path: Path) -> None:
    """export() must not leave the model in eval mode or move it permanently."""
    from getitune.types.export import ExportFormat

    model = HFMulticlassClsModel(_tiny_vit_config(), _label_info(), data_input_params={"input_size": (224, 224)})
    model.train()
    original_forward = model.forward

    model.export(tmp_path, "exported_model", ExportFormat.ONNX)

    assert model.training is True
    assert model.forward == original_forward


def test_dispatch_label_info_accepts_plain_int() -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), 5)
    assert model.label_info.num_classes == 5


def test_dispatch_label_info_accepts_name_list() -> None:
    model = HFMulticlassClsModel(_tiny_vit_config(), ["a", "b"])
    assert model.label_info.label_names == ["a", "b"]


def test_dispatch_label_info_rejects_unknown_type() -> None:
    with pytest.raises(TypeError):
        HFModel._dispatch_label_info(object())  # type: ignore[arg-type]


def test_dispatch_label_info_passes_through_seg_label_info_subclass() -> None:
    seg_label_info = SegLabelInfo(label_names=["a", "b"], label_ids=["0", "1"], label_groups=[["a", "b"]])
    assert HFModel._dispatch_label_info(seg_label_info) is seg_label_info
