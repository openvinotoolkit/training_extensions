# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the Ultralytics engine workflow.

Exercises the end-to-end getitune workflow with Ultralytics-backed models:

1. Instantiate ``UltralyticsEngine`` via Configurator + create_engine().
2. Train for 2 epochs.
3. Evaluate the model (``engine.test``).
4. Get predictions (``engine.predict``).
5. Export to OpenVINO IR and validate with ModelAPI.
6. Evaluate with OVEngine (``ov_engine.test``).
7. Get OV predictions (``ov_engine.predict``).
8. Quantize to INT8 (``ov_engine.optimize``) and validate.

Uses parquet-based datasets under ``tests/assets/``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Iterator, NamedTuple, cast

import numpy as np
import pytest
from jsonargparse import ArgumentParser, Namespace
from model_api.models import Model

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.openvino.engine import OVEngine
from getitune.backend.ultralytics.engine import UltralyticsEngine
from getitune.backend.ultralytics.models.base import UltralyticsModel
from getitune.backend.ultralytics.tools.configurator import Configurator
from getitune.config.data import SamplerConfig, SubsetConfig, TileConfig
from getitune.data.module import DataModule
from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.types.task import TaskType

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"
RECIPE_ROOT = Path(__file__).resolve().parents[3] / "src" / "getitune" / "recipe"
logger = logging.getLogger(__name__)


class _TaskSpec(NamedTuple):
    task: TaskType
    recipe_name: str
    dataset_dir: str


_TASK_SPECS: list[_TaskSpec] = [
    _TaskSpec(
        task=TaskType.DETECTION,
        recipe_name="yolo26_n",
        dataset_dir="detection_coco",
    ),
    _TaskSpec(
        task=TaskType.INSTANCE_SEGMENTATION,
        recipe_name="yolo26_n_seg",
        dataset_dir="instance_segmentation_coco",
    ),
    _TaskSpec(
        task=TaskType.MULTI_CLASS_CLS,
        recipe_name="yolo26_n_cls",
        dataset_dir="classification_cifar10",
    ),
    _TaskSpec(
        task=TaskType.MULTI_LABEL_CLS,
        recipe_name="yolo26_n_cls",
        dataset_dir="multilabel_classification_coco",
    ),
    _TaskSpec(
        task=TaskType.SEMANTIC_SEGMENTATION,
        recipe_name="yolo26_n_sem",
        dataset_dir="segmentation_pets",
    ),
]


def _resolve_recipe(spec: _TaskSpec) -> Path:
    task_to_subdir = {
        TaskType.DETECTION: "detection",
        TaskType.INSTANCE_SEGMENTATION: "instance_segmentation",
        TaskType.MULTI_CLASS_CLS: "classification/multi_class_cls",
        TaskType.MULTI_LABEL_CLS: "classification/multi_label_cls",
        TaskType.SEMANTIC_SEGMENTATION: "semantic_segmentation",
    }
    subdir = task_to_subdir[spec.task]
    return RECIPE_ROOT / subdir / f"{spec.recipe_name}.yaml"


def _id_fn(spec: _TaskSpec) -> str:
    return f"{spec.task.value}-{spec.recipe_name}"


_FILTERED_TASK_SPECS: list[_TaskSpec] = [
    spec for spec in _TASK_SPECS if spec.task in getattr(pytest, "TASK_LIST", list(TaskType))
]


def _build_subset_config(data_config: dict, subset_name: str) -> SubsetConfig:
    """Build a SubsetConfig from the recipe's data config."""
    subset_cfg_data = deepcopy(data_config[f"{subset_name}_subset"])
    subset_cfg_data["input_size"] = data_config["input_size"]
    sampler_cfg_data = subset_cfg_data.pop("sampler", {})
    return SubsetConfig(sampler=SamplerConfig(**sampler_cfg_data), **subset_cfg_data)


@contextmanager
def _timed_step(spec: _TaskSpec, step_name: str) -> Iterator[None]:
    """Log step start/end with execution time for workflow visibility."""
    start = perf_counter()
    logger.info("[%s] %s started", spec.task.value, step_name)
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("[%s] %s finished in %.2fs", spec.task.value, step_name, elapsed)


@pytest.mark.parametrize("spec", _FILTERED_TASK_SPECS, ids=_id_fn)
def test_ultralytics_engine_workflow(
    spec: _TaskSpec,
    tmp_path: Path,
    fxt_accelerator: str,
) -> None:
    """End-to-end Ultralytics engine workflow: train -> test -> predict -> export -> OV test -> quantize."""
    data_root = ASSETS_ROOT / spec.dataset_dir
    if not data_root.exists():
        pytest.skip(f"Dataset not found at {data_root}")

    recipe = _resolve_recipe(spec)
    if not recipe.exists():
        pytest.skip(f"Recipe not found: {recipe}")

    work_dir = tmp_path / spec.task.value

    # Parse recipe to a config dict
    config = Configurator.convert(recipe)
    data_config = config["data"]
    training_config = config.get("training", {})

    train_subset = _build_subset_config(data_config, "train")
    val_subset = _build_subset_config(data_config, "val")
    test_subset = _build_subset_config(data_config, "test")

    datamodule = DataModule(
        task=spec.task,
        data_root=str(data_root),
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        tile_config=TileConfig(enable_tiler=False),
        input_size=tuple(data_config["input_size"]),
    )

    model_cfg = deepcopy(config["model"])
    model_cfg["init_args"]["label_info"] = datamodule.label_info.label_names
    model_cfg["init_args"]["data_input_params"] = DataInputParams(
        input_size=cast("tuple[int, int]", datamodule.input_size),
        mean=datamodule.input_mean if datamodule.input_mean is not None else (0.0, 0.0, 0.0),
        std=datamodule.input_std if datamodule.input_std is not None else (1.0, 1.0, 1.0),
        intensity_config=datamodule.input_intensity_config,
    ).as_dict()

    model_parser = ArgumentParser()
    model_parser.add_argument("--model", type=UltralyticsModel)
    model = model_parser.instantiate_classes(Namespace(model=model_cfg)).get("model")
    assert isinstance(model, UltralyticsModel)

    engine = create_engine(
        model=model,
        data=datamodule,
        work_dir=work_dir,
        device="auto",
        train_args=training_config,
        export_args={
            "confidence_threshold": config.get("export", {}).get("confidence_threshold", 0.25),
            "iou_threshold": config.get("export", {}).get("iou_threshold", 0.5),
        },
    )
    assert isinstance(engine, UltralyticsEngine)

    with _timed_step(spec, "train"):
        train_metrics = engine.train(max_epochs=2)
    assert len(train_metrics) > 0

    with _timed_step(spec, "test"):
        test_metrics = engine.test()
    assert len(test_metrics) > 0

    with _timed_step(spec, "predict"):
        predictions = engine.predict()
    assert predictions is not None
    assert len(predictions) > 0

    with _timed_step(spec, "export_openvino_fp32"):
        ov_xml_path = engine.export(
            export_format=ExportFormat.OPENVINO,
            export_precision=Precision.FP32,
        )
    assert ov_xml_path.exists()
    assert ov_xml_path.suffix == ".xml"

    assert datamodule.input_size is not None
    input_h, input_w = datamodule.input_size
    dummy_input = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    with _timed_step(spec, "model_api_fp32_create"):
        mapi_model = Model.create_model(str(ov_xml_path))
    assert mapi_model is not None
    with _timed_step(spec, "model_api_fp32_infer"):
        fp32_result = mapi_model(dummy_input)
    assert fp32_result is not None

    ov_engine = create_engine(
        model=ov_xml_path,
        data=datamodule,
        work_dir=str(work_dir / "ov"),
    )
    assert isinstance(ov_engine, OVEngine)

    with _timed_step(spec, "ov_test"):
        ov_test_metrics = ov_engine.test()
    assert len(ov_test_metrics) > 0

    with _timed_step(spec, "ov_predict"):
        ov_predictions = ov_engine.predict()
    assert ov_predictions is not None
    assert len(ov_predictions) > 0

    with _timed_step(spec, "ov_optimize"):
        optimized_path = ov_engine.optimize()
    assert optimized_path.exists()

    with _timed_step(spec, "model_api_int8_create"):
        mapi_int8_model = Model.create_model(str(optimized_path))
    assert mapi_int8_model is not None
    with _timed_step(spec, "model_api_int8_infer"):
        int8_result = mapi_int8_model(dummy_input)
    assert int8_result is not None
