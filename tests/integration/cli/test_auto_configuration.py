# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

from tests.integration.cli.utils import run_main


def test_otx_cli_auto_configuration(
    task: OTXTaskType,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
) -> None:
    """Test the OTX auto configuration with CLI.

    Args:
        task (OTXTaskType): The task to be performed.
        tmp_path (Path): The temporary path for storing outputs.
        fxt_accelerator (str): The accelerator to be used.
        fxt_target_dataset_per_task (dict): The target dataset per task.

    Returns:
        None
    """
    if task not in DEFAULT_CONFIG_PER_TASK:
        pytest.skip(f"Task {task} is not supported in the auto-configuration.")
    if task.lower() in ("action_classification"):
        pytest.xfail(reason="xFail until this root cause is resolved on the Datumaro side.")
    tmp_path_train = tmp_path / f"otx_auto_train_{task}"
    command_cfg = [
        "otx",
        "train",
        "--data_root",
        fxt_target_dataset_per_task[task.lower()],
        "--task",
        task.upper(),
        "--engine.work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        *fxt_cli_override_command_per_task[task.lower()],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    # Currently, a simple output check
    assert (tmp_path_train / "outputs").exists()
    assert (tmp_path_train / "outputs" / "configs.yaml").exists()
    assert (tmp_path_train / "outputs" / "csv").exists()
    assert (tmp_path_train / "outputs" / "checkpoints").exists()
    ckpt_files = list((tmp_path_train / "outputs" / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0
