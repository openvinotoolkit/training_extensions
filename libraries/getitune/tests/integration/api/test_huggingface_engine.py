# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Hugging Face engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pytest
from model_api.models import Model

from getitune.backend.huggingface.engine import HFEngine
from getitune.backend.openvino.engine import OVEngine
from getitune.data import DataModule
from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.types.task import TaskType

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"
RECIPE_ROOT = Path(__file__).resolve().parents[3] / "src" / "getitune" / "recipe"


class _TaskSpec(NamedTuple):
    task: TaskType
    recipe_name: str
    dataset_dir: str


_TASK_SPECS = [
    _TaskSpec(TaskType.DETECTION, "rtdetrv2_r18", "detection_coco"),
    _TaskSpec(TaskType.INSTANCE_SEGMENTATION, "mask2former_swin_t", "instance_segmentation_coco"),
    _TaskSpec(TaskType.SEMANTIC_SEGMENTATION, "segformer_b0", "segmentation_pets"),
    _TaskSpec(TaskType.MULTI_CLASS_CLS, "vit_b16", "classification_cifar10"),
    _TaskSpec(TaskType.MULTI_LABEL_CLS, "vit_b16", "multilabel_classification_coco"),
]


def _resolve_recipe(spec: _TaskSpec) -> Path:
    task_to_subdir = {
        TaskType.DETECTION: "detection",
        TaskType.INSTANCE_SEGMENTATION: "instance_segmentation",
        TaskType.SEMANTIC_SEGMENTATION: "semantic_segmentation",
        TaskType.MULTI_CLASS_CLS: "classification/multi_class_cls",
        TaskType.MULTI_LABEL_CLS: "classification/multi_label_cls",
    }
    return RECIPE_ROOT / task_to_subdir[spec.task] / f"{spec.recipe_name}.yaml"


def _id_fn(spec: _TaskSpec) -> str:
    return spec.task.value


_FILTERED_TASK_SPECS = [spec for spec in _TASK_SPECS if spec.task in getattr(pytest, "TASK_LIST", list(TaskType))]


@pytest.mark.parametrize("spec", _FILTERED_TASK_SPECS, ids=_id_fn)
def test_huggingface_engine_workflow(spec: _TaskSpec, tmp_path: Path, fxt_accelerator: str) -> None:
    data_root = ASSETS_ROOT / spec.dataset_dir
    recipe = _resolve_recipe(spec)
    if not data_root.exists():
        pytest.skip(f"Dataset not found at {data_root}")
    if not recipe.exists():
        pytest.skip(f"Recipe not found at {recipe}")

    engine_kwargs: dict[str, Any] = {
        "model": recipe,
        "data": data_root,
        "work_dir": tmp_path / spec.task.value,
        "device": fxt_accelerator,
        "pretrained": False,
    }
    if spec.task == TaskType.INSTANCE_SEGMENTATION:
        pytest.skip(
            reason="OOM on CI"
        )  # TODO @kprokofi: Remove this skip once OOM issue is resolved (use smaller model once available)
        engine_kwargs["input_size"] = (512, 512)

    engine = create_engine(
        **engine_kwargs,
    )
    assert isinstance(engine, HFEngine)

    train_metrics = engine.train(max_epochs=1, batch=1, precision="32")
    assert train_metrics

    test_metrics = engine.test(batch=1)
    assert test_metrics
    assert all(key.startswith("test/") for key in test_metrics)

    predictions = engine.predict(batch=1)
    assert len(predictions) == len(cast("DataModule", engine.datamodule).subsets["test"])

    onnx_path = engine.export(export_format=ExportFormat.ONNX, export_precision=Precision.FP32)
    assert onnx_path.exists()
    assert Model.create_model(str(onnx_path)) is not None

    onnx_fp16_path = engine.export(export_format=ExportFormat.ONNX, export_precision=Precision.FP16)
    assert onnx_fp16_path.exists()
    assert Model.create_model(str(onnx_fp16_path)) is not None

    openvino_path = engine.export(export_format=ExportFormat.OPENVINO, export_precision=Precision.FP32)
    assert openvino_path.exists()
    assert openvino_path.suffix == ".xml"

    input_size = cast("DataModule", engine.datamodule).input_size
    assert input_size is not None
    input_h, input_w = input_size
    dummy_input = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    model_api = Model.create_model(str(openvino_path))
    assert model_api is not None
    assert model_api(dummy_input) is not None

    openvino_fp16_path = engine.export(export_format=ExportFormat.OPENVINO, export_precision=Precision.FP16)
    assert openvino_fp16_path.exists()
    assert openvino_fp16_path.suffix == ".xml"
    model_api_fp16 = Model.create_model(str(openvino_fp16_path))
    assert model_api_fp16 is not None
    assert model_api_fp16(dummy_input) is not None

    ov_engine = create_engine(
        model=openvino_fp16_path,
        data=engine.datamodule,
        work_dir=tmp_path / f"{spec.task.value}_openvino",
    )
    assert isinstance(ov_engine, OVEngine)
    assert ov_engine.test(batch=1)
    assert ov_engine.predict(batch=1)
