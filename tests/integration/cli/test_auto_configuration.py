# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from unittest.mock import patch

import pytest
from otx.cli import main
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK


@pytest.mark.parametrize("task", [task.value.lower() for task in DEFAULT_CONFIG_PER_TASK])
def test_otx_cli_auto_configuration(
    task: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
) -> None:
    """Test the OTX auto configuration with CLI.

    Args:
        task (str): The task to be performed.
        tmp_path (Path): The temporary path for storing outputs.
        fxt_accelerator (str): The accelerator to be used.
        fxt_target_dataset_per_task (dict): The target dataset per task.

    Returns:
        None
    """
    if task in ("action_classification"):
        pytest.xfail(reason="xFail until this root cause is resolved on the Datumaro side.")
    tmp_path_train = tmp_path / f"otx_auto_train_{task}"
    command_cfg = [
        "otx",
        "train",
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--task",
        task.upper(),
        "--engine.work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        *fxt_cli_override_command_per_task[task],
    ]

    with patch("sys.argv", command_cfg):
        main()

    # Currently, a simple output check
    assert (tmp_path_train / "outputs").exists()
    assert (tmp_path_train / "outputs" / "configs.yaml").exists()
    assert (tmp_path_train / "outputs" / "csv").exists()
    assert (tmp_path_train / "outputs" / "checkpoints").exists()
    ckpt_files = list((tmp_path_train / "outputs" / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0
