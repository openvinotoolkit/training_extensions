# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.engine import Engine
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK


@pytest.mark.parametrize("task", pytest.TASK_LIST)
def test_auto_configuration(
    task: OTXTaskType,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
) -> None:
    """Test the auto configuration functionality.

    Args:
        task (OTXTaskType): The task for which auto configuration is being tested.
        tmp_path (Path): The temporary path for storing training data.
        fxt_accelerator (str): The accelerator used for training.
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
    """
    if task not in DEFAULT_CONFIG_PER_TASK:
        pytest.skip(f"Task {task} is not supported in the auto-configuration.")
    if task.lower() in ("h_label_cls"):
        pytest.skip(
            reason="H-labels require num_multiclass_head, num_multilabel_classes, which skip until we have the ability to automate this.",
        )
    if task.lower().startswith("anomaly"):
        pytest.skip(reason="This will be added in a future pipeline behavior.")

    tmp_path_train = tmp_path / f"auto_train_{task}"
    data_root = fxt_target_dataset_per_task[task.lower()]
    engine = Engine(
        data_root=data_root,
        task=task,
        work_dir=tmp_path_train,
        device=fxt_accelerator,
    )
    if task.lower() == "zero_shot_visual_prompting":
        engine.model.infer_reference_info_root = Path(tmp_path_train)
        # update litmodule.hparams to reflect changed hparams
        engine.model.hparams.update({"infer_reference_info_root": str(engine.model.infer_reference_info_root)})

    # Check OTXModel & OTXDataModule
    assert isinstance(engine.model, OTXModel)
    assert isinstance(engine.datamodule, OTXDataModule)

    # Check Auto-Configurator task
    assert engine._auto_configurator.task == task

    # Check Default Configuration
    from otx.cli.utils.jsonargparse import get_configuration

    default_config = get_configuration(DEFAULT_CONFIG_PER_TASK[task])
    default_config["data"]["data_root"] = data_root
    num_classes = engine.datamodule.label_info.num_classes

    default_config["model"]["init_args"]["num_classes"] = num_classes

    drop_model_config = lambda cfg: {key: value for key, value in cfg.items() if key != "model"}
    assert drop_model_config(engine._auto_configurator.config) == drop_model_config(default_config)

    max_epochs = 2 if task.lower() != "zero_shot_visual_prompting" else 1
    train_metric = engine.train(max_epochs=max_epochs)
    if task.lower() != "zero_shot_visual_prompting":
        assert len(train_metric) > 0

    test_metric = engine.test()
    assert len(test_metric) > 0
