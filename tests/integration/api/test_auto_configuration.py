# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.engine import Engine
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK, get_num_classes_from_meta_info


@pytest.mark.parametrize("task", [task.value.lower() for task in DEFAULT_CONFIG_PER_TASK])
def test_auto_configuration(task: str, tmp_path: Path, fxt_accelerator: str, fxt_target_dataset_per_task: dict) -> None:
    """Test the auto configuration functionality.

    Args:
        task (str): The task for which auto configuration is being tested.
        tmp_path (Path): The temporary path for storing training data.
        fxt_accelerator (str): The accelerator used for training.
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
    """
    tmp_path_train = tmp_path / f"auto_train_{task}"
    data_root = fxt_target_dataset_per_task[task]
    task_type = OTXTaskType(task.upper())
    engine = Engine(
        data_root=data_root,
        task=task_type,
        work_dir=tmp_path_train,
        device=fxt_accelerator,
    )

    # Check OTXModel & OTXDataModule
    assert isinstance(engine.model, OTXModel)
    assert isinstance(engine.datamodule, OTXDataModule)

    # Check Auto-Configurator task
    assert engine._auto_configurator.task == task_type

    # Check Default Configuration
    from otx.cli.utils.jsonargparse import get_configuration

    default_config = get_configuration(DEFAULT_CONFIG_PER_TASK[task_type])
    default_config["data"]["config"]["data_root"] = data_root
    num_classes = get_num_classes_from_meta_info(task=task_type, meta_info=engine.datamodule.meta_info)

    default_config["model"]["init_args"]["num_classes"] = num_classes

    assert engine._auto_configurator.config == default_config

    train_metric = engine.train(max_epochs=default_config.get("max_epochs", 2))
    if task != "zero_shot_visual_prompting":
        assert len(train_metric) > 0

    test_metric = engine.test()
    assert len(test_metric) > 0
