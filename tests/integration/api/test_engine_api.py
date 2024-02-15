# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OTXModel
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

    train_metric = engine.train(max_epochs=2)
    if task.lower() != "zero_shot_visual_prompting":
        assert len(train_metric) > 0

    test_metric = engine.test()
    assert len(test_metric) > 0

    # A Task that doesn't have Export implemented yet.
    # [TODO]: Enable should progress for all Tasks.
    if task in [
        OTXTaskType.ACTION_CLASSIFICATION,
        OTXTaskType.ACTION_DETECTION,
        OTXTaskType.H_LABEL_CLS,
        OTXTaskType.ROTATED_DETECTION,
        OTXTaskType.VISUAL_PROMPTING,
        OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING,
    ]:
        return

    # Export IR Model
    exported_model_path = engine.export()
    assert exported_model_path.exists()

    # Test with IR Model
    if task in OVMODEL_PER_TASK:
        test_metric_from_ov_model = engine.test(checkpoint=exported_model_path, accelerator="cpu")
        assert len(test_metric_from_ov_model) > 0
