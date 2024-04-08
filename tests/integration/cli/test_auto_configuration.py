# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

from tests.utils import run_main


@pytest.mark.parametrize("task", pytest.TASK_LIST)
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
    tmp_path_train = tmp_path / f"otx_auto_train_{task}"
    command_cfg = [
        "otx",
        "train",
        "--data_root",
        fxt_target_dataset_per_task[task.lower()],
        "--task",
        task.upper(),
        "--work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "1" if task.lower() in ("zero_shot_visual_prompting") else "2",
        *fxt_cli_override_command_per_task[task.lower()],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    # Currently, a simple output check
    outputs_dir = tmp_path_train / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    assert latest_dir.exists()
    assert (latest_dir / "configs.yaml").exists()
    assert (latest_dir / "csv").exists()
    assert (latest_dir / "checkpoints").exists()
    ckpt_files = list((latest_dir / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0
