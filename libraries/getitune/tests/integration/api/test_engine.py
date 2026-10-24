# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the core getitune engine workflow.

This single test module replaces the old ``test_engine_api.py`` and
``test_geti_interaction.py``.  It is parametrised to cover one
(small / fast) model architecture per task and exercises the typical
end-to-end getitune workflow:

1. Instantiate ``LightningEngine`` from a recipe config + data root.
2. Train for 1 epoch.
3. Evaluate the model (``engine.test``).
4. Get predictions (``engine.predict``).
5. Export to ONNX and predict with the exported model.
6. Export to OpenVINO IR and predict with the exported model.
7. Quantize the model to INT8 (``ov_engine.optimize``) and predict with it.

The test uses the *new Datumaro experimental format* (parquet-based
datasets under ``tests/assets/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from model_api.models import Model

from getitune.backend.lightning.engine import LightningEngine
from getitune.backend.lightning.models.base import LightningModel
from getitune.backend.openvino.engine import OVEngine
from getitune.data.module import DataModule
from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.types.task import TaskType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"


class _TaskSpec(NamedTuple):
    """Light-weight descriptor for a (task, recipe, dataset) triple."""

    task: TaskType
    recipe_name: str  # e.g. "mobilenet_v3_large"
    dataset_dir: str  # relative to ASSETS_ROOT


# One *small* model per task - chosen for fast training.
_TASK_SPECS: list[_TaskSpec] = [
    _TaskSpec(
        task=TaskType.MULTI_CLASS_CLS,
        recipe_name="mobilenet_v3_large",
        dataset_dir="classification_cifar10",
    ),
    _TaskSpec(
        task=TaskType.MULTI_LABEL_CLS,
        recipe_name="mobilenet_v3_large",
        dataset_dir="multilabel_classification_coco",
    ),
    _TaskSpec(
        task=TaskType.DETECTION,
        recipe_name="ssd_mobilenetv2",
        dataset_dir="detection_coco",
    ),
    _TaskSpec(
        task=TaskType.INSTANCE_SEGMENTATION,
        recipe_name="rtmdet_inst_tiny",
        dataset_dir="instance_segmentation_coco",
    ),
    _TaskSpec(
        task=TaskType.SEMANTIC_SEGMENTATION,
        recipe_name="litehrnet_s",
        dataset_dir="segmentation_pets",
    ),
    _TaskSpec(
        task=TaskType.KEYPOINT_DETECTION,
        recipe_name="rtmpose_tiny",
        dataset_dir="keypoint_detection_coco",
    ),
    _TaskSpec(
        task=TaskType.MULTI_CLASS_CLS,
        recipe_name="efficientnet_b0",
        dataset_dir="classification_dataset_16bit",
    ),
]


def _resolve_recipe(spec: _TaskSpec) -> str:
    """Return the absolute recipe YAML path for a given task spec."""
    from getitune.tools.auto_configurator import RECIPE_PATH

    # Map task enum to the recipe subdirectory
    task_to_subdir = {
        TaskType.MULTI_CLASS_CLS: "classification/multi_class_cls",
        TaskType.MULTI_LABEL_CLS: "classification/multi_label_cls",
        TaskType.DETECTION: "detection",
        TaskType.INSTANCE_SEGMENTATION: "instance_segmentation",
        TaskType.SEMANTIC_SEGMENTATION: "semantic_segmentation",
        TaskType.KEYPOINT_DETECTION: "keypoint_detection",
    }
    subdir = task_to_subdir[spec.task]
    recipe_path = RECIPE_PATH / subdir / f"{spec.recipe_name}.yaml"
    if not recipe_path.exists():
        msg = f"Recipe not found: {recipe_path}"
        raise FileNotFoundError(msg)
    return str(recipe_path)


def _id_fn(spec: _TaskSpec) -> str:
    """Readable test-ID for ``pytest.mark.parametrize``."""
    base = f"{spec.task.value}-{spec.recipe_name}"
    if "16bit" in spec.dataset_dir:
        return f"{base}-16bit"
    return base


# Filter specs based on ``--task`` CLI option (populated via ``pytest.TASK_LIST``
# in ``conftest.py``).  When ``--task all`` (the default) every spec is kept;
# otherwise only specs whose task appears in the requested list are executed.
_FILTERED_TASK_SPECS: list[_TaskSpec] = [
    spec for spec in _TASK_SPECS if spec.task in getattr(pytest, "TASK_LIST", list(TaskType))
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", _FILTERED_TASK_SPECS, ids=_id_fn)
def test_engine_workflow(
    spec: _TaskSpec,
    tmp_path: Path,
    fxt_accelerator: str,
) -> None:
    """End-to-end engine workflow: train ➜ test ➜ predict ➜ export ➜ OV predict ➜ quantize.

    Args:
        spec: Task specification (task type, recipe name, dataset directory).
        tmp_path: Pytest-provided temporary directory.
        fxt_accelerator: Device accelerator fixture (``"gpu"`` / ``"cpu"``).
    """
    data_root = ASSETS_ROOT / spec.dataset_dir
    if not data_root.exists():
        pytest.skip(f"Benchmark dataset not found at {data_root}. Run the corresponding download script first.")

    recipe = _resolve_recipe(spec)
    work_dir = tmp_path / spec.task.value

    # ---- 1. Instantiate engine ------------------------------------------
    engine = LightningEngine.from_config(
        config_path=recipe,
        data=str(data_root),
        work_dir=str(work_dir),
        device=fxt_accelerator,
    )
    assert isinstance(engine.model, LightningModel)
    assert isinstance(engine.datamodule, DataModule)

    # ---- 2. Train for a minimal number of epochs -----------------------
    train_metrics = engine.train(max_epochs=1, precision="32")
    assert len(train_metrics) > 0
    assert engine.checkpoint is not None

    # ---- 3. Test (evaluate) --------------------------------------------
    test_metrics = engine.test()
    assert len(test_metrics) > 0

    # ---- 4. Predict (torch model) --------------------------------------
    predictions = engine.predict()
    assert predictions is not None
    assert len(predictions) > 0

    # ---- 5. Export to ONNX & predict -----------------------------------
    onnx_path = engine.export(
        export_format=ExportFormat.ONNX,
        export_precision=Precision.FP32,
    )
    assert onnx_path.exists()
    assert onnx_path.suffix == ".onnx"

    # ModelAPI loads hierarchical classification models via GreedyLabelsResolver,
    # which requires all parent nodes to be present in label_to_idx.
    mapi_onnx_model = Model.create_model(str(onnx_path))
    assert mapi_onnx_model is not None

    # Run a quick inference to validate the exported ONNX model is functional
    dummy_input = np.zeros((224, 224, 3), dtype=np.uint8)
    onnx_result = mapi_onnx_model(dummy_input)
    assert onnx_result is not None

    # ---- 6. Export to OpenVINO IR & predict with OVEngine ---------------
    ov_xml_path = engine.export(
        export_format=ExportFormat.OPENVINO,
        export_precision=Precision.FP32,
    )
    assert ov_xml_path.exists()
    assert ov_xml_path.suffix == ".xml"

    # For 16-bit datasets, verify that intensity config is properly embedded
    # in the exported model's rt_info metadata.
    if "16bit" in spec.dataset_dir:
        import openvino

        ov_model = openvino.Core().read_model(str(ov_xml_path))
        assert ov_model.get_rt_info(["model_info", "input_dtype"]).astype(str) == "u16"
        assert ov_model.get_rt_info(["model_info", "intensity_mode"]).astype(str) == "scale_to_unit"

    ov_engine = create_engine(
        model=ov_xml_path,
        data=engine.datamodule,
        work_dir=str(work_dir / "ov"),
    )
    assert isinstance(ov_engine, OVEngine)

    ov_test_metrics = ov_engine.test()
    assert len(ov_test_metrics) > 0

    ov_predictions = ov_engine.predict()
    assert ov_predictions is not None
    assert len(ov_predictions) > 0

    # ---- 7. Quantize to INT8 & predict ---------------------------------
    optimized_path = ov_engine.optimize()
    assert optimized_path.exists()

    mapi_int8_model = Model.create_model(str(optimized_path))
    assert mapi_int8_model is not None

    # Run a quick inference to validate the quantized INT8 model is functional
    int8_result = mapi_int8_model(dummy_input)
    assert int8_result is not None
