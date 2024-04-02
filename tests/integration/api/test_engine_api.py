# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from openvino.model_api.tilers import Tiler
from otx.algo.classification.efficientnet_b0 import EfficientNetB0ForMulticlassCls
from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.engine import Engine
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK, OVMODEL_PER_TASK


@pytest.mark.parametrize("task", pytest.TASK_LIST)
def test_engine_from_config(
    task: OTXTaskType,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
) -> None:
    """Test the Engine.from_config functionality.

    Args:
        task (OTXTaskType): The task type.
        tmp_path (Path): The temporary path for storing training data.
        fxt_accelerator (str): The accelerator used for training.
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
    """
    if task not in DEFAULT_CONFIG_PER_TASK:
        pytest.skip("Only the Task has Default config is tested to reduce unnecessary resources.")
    if task.lower() in ("action_classification"):
        pytest.xfail(reason="xFail until this root cause is resolved on the Datumaro side.")
    if task.lower() in ("h_label_cls"):
        pytest.skip(
            reason="H-labels require num_multiclass_head, num_multilabel_classes, which skip until we have the ability to automate this.",
        )

    tmp_path_train = tmp_path / task
    engine = Engine.from_config(
        config_path=DEFAULT_CONFIG_PER_TASK[task],
        data_root=fxt_target_dataset_per_task[task.value.lower()],
        work_dir=tmp_path_train,
        device=fxt_accelerator,
    )

    # Check OTXModel & OTXDataModule
    assert isinstance(engine.model, OTXModel)
    assert isinstance(engine.datamodule, OTXDataModule)

    max_epochs = 2 if task.lower() != "zero_shot_visual_prompting" else 1
    train_metric = engine.train(max_epochs=max_epochs)
    if task.lower() != "zero_shot_visual_prompting":
        assert len(train_metric) > 0

    test_metric = engine.test()
    assert len(test_metric) > 0

    predict_result = engine.predict()
    assert len(predict_result) > 0

    # A Task that doesn't have Export implemented yet.
    # [TODO]: Enable should progress for all Tasks.
    if task in [
        OTXTaskType.ACTION_CLASSIFICATION,
        OTXTaskType.ACTION_DETECTION,
        OTXTaskType.H_LABEL_CLS,
        OTXTaskType.ROTATED_DETECTION,
        OTXTaskType.ANOMALY_CLASSIFICATION,
        OTXTaskType.ANOMALY_DETECTION,
        OTXTaskType.ANOMALY_SEGMENTATION,
    ]:
        return

    # Export IR Model
    exported_model_path: Path | dict[str, Path] = engine.export()
    if isinstance(exported_model_path, Path):
        assert exported_model_path.exists()
    elif isinstance(exported_model_path, dict):
        for key, value in exported_model_path.items():
            assert value.exists(), f"{value} for {key} doesn't exist."
    else:
        AssertionError(f"Exported model path is not a Path or a dictionary of Paths: {exported_model_path}")

    # Test with IR Model
    if task in OVMODEL_PER_TASK:
        if task.lower() in ["visual_prompting", "zero_shot_visual_prompting"]:
            test_metric_from_ov_model = engine.test(checkpoint=exported_model_path["decoder"], accelerator="cpu")
        else:
            test_metric_from_ov_model = engine.test(checkpoint=exported_model_path, accelerator="cpu")
        assert len(test_metric_from_ov_model) > 0

    # List of models with explain supported.
    if task not in [
        OTXTaskType.MULTI_CLASS_CLS,
        OTXTaskType.MULTI_LABEL_CLS,
        # Restore these models after fixing undetermined CI failures for ATSS and Mask RCNN
        # OTXTaskType.DETECTION,
        # OTXTaskType.ROTATED_DETECTION,
        # OTXTaskType.INSTANCE_SEGMENTATION,
    ]:
        return

    # Predict Torch model with explain
    predictions = engine.predict(explain=True)
    assert len(predictions[0].saliency_maps) > 0

    # Export IR model with explain
    exported_model_with_explain = engine.export(explain=True)
    assert exported_model_with_explain.exists()

    # Infer IR Model with explain: predict
    predictions = engine.predict(explain=True, checkpoint=exported_model_with_explain, accelerator="cpu")
    assert len(predictions) > 0
    sal_maps_from_prediction = predictions[0].saliency_maps
    assert len(sal_maps_from_prediction) > 0

    # Infer IR Model with explain: explain
    explain_results = engine.explain(checkpoint=exported_model_with_explain, accelerator="cpu")
    assert len(explain_results[0].saliency_maps) > 0
    sal_maps_from_explain = explain_results[0].saliency_maps
    assert (sal_maps_from_prediction[0][0] == sal_maps_from_explain[0][0]).all()


@pytest.mark.parametrize("recipe", pytest.TILE_RECIPE_LIST)
def test_engine_from_tile_recipe(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
):
    task = OTXTaskType.DETECTION if "detection" in recipe else OTXTaskType.INSTANCE_SEGMENTATION

    engine = Engine.from_config(
        config_path=recipe,
        data_root=fxt_target_dataset_per_task[task.value.lower()],
        work_dir=tmp_path / task,
        device=fxt_accelerator,
    )
    engine.train(max_epochs=1)
    exported_model_path = engine.export()
    assert exported_model_path.exists()
    metric = engine.test(exported_model_path, accelerator="cpu")
    assert len(metric) > 0

    # Check OVModel & OVTiler is set correctly
    ov_model = engine._auto_configurator.get_ov_model(
        model_name=exported_model_path,
        label_info=engine.datamodule.label_info,
    )
    assert isinstance(ov_model.model, Tiler), "Model should be an instance of Tiler"
    assert engine.datamodule.config.tile_config.tile_size[0] == ov_model.model.tile_size
    assert engine.datamodule.config.tile_config.overlap == ov_model.model.tiles_overlap


REASON = """
Traceback (most recent call last):
  File "/home/vinnamki/miniconda3/envs/otx-v2/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/home/vinnamki/miniconda3/envs/otx-v2/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/vinnamki/otx/training_extensions/src/otx/hpo/hpo_runner.py", line 200, in _run_train
    train_func(hp_config, report_func)
  File "/home/vinnamki/otx/training_extensions/src/otx/engine/hpo/hpo_trial.py", line 75, in run_hpo_trial
    callbacks = _register_hpo_callback(report_func, callbacks)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vinnamki/otx/training_extensions/src/otx/engine/hpo/hpo_trial.py", line 101, in _register_hpo_callback
    callbacks.append(HPOCallback(report_func, _get_metric(callbacks)))
                                              ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vinnamki/otx/training_extensions/src/otx/engine/hpo/hpo_trial.py", line 110, in _get_metric
    raise RuntimeError(error_msg)
RuntimeError: Failed to find a metric. There is no ModelCheckpoint in callback list.
"""


@pytest.mark.parametrize("task", pytest.TASK_LIST)
def test_otx_hpo(
    task: OTXTaskType,
    tmp_path: Path,
    fxt_target_dataset_per_task: dict,
) -> None:
    pytest.xfail(reason=REASON)

    model = EfficientNetB0ForMulticlassCls(num_classes=3)
    work_dir = str(tmp_path)
    engine = Engine(
        data_root=fxt_target_dataset_per_task[task.lower()],
        task=task,
        work_dir=work_dir,
        model=model,
    )
    engine.train(run_hpo=True)
