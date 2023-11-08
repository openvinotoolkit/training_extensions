# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from tests.v2.integration.cli.test_helper import check_run
from tests.v2.integration.test_helper import TASK_CONFIGURATION


@pytest.mark.parametrize("task", TASK_CONFIGURATION.keys())
def test_otx_cli(task: str, tmp_dir_path: Path) -> None:
    """
    End-to-end test for the otx CLI tool. This test covers the following steps:
    1. Training
    2. Training with config
    3. Testing
    4. Prediction with single image
    5. Export Openvino IR Model

    Args:
        task (str): The name of the task to test.
        tmp_dir_path (Path): The path to the temporary directory to use for testing.

    Returns:
        None
    """
    # 1. Training
    tmp_dir_path = tmp_dir_path / f"{task}_e2e_test"
    deterministic = "False" if task == "segmentation" else "True"
    command_line = [
        "otx", "train",
        "--work_dir", str(tmp_dir_path),
        "--data.task", task,
        "--data.train_data_roots", TASK_CONFIGURATION[task]["train_data_roots"],
        "--data.val_data_roots", TASK_CONFIGURATION[task]["val_data_roots"],
        "--max_epochs", "1",
        "--seed", "1234",
        "--deterministic", deterministic,
    ]
    rc, stdout, _ = check_run(command_line)
    assert rc == 0
    assert "time elapsed" in str(stdout)
    checkpoint_path = tmp_dir_path / f"{task}" / "latest" / "weights.pth"
    config_path = tmp_dir_path / f"{task}" / "latest" / "configs.yaml"
    assert checkpoint_path.exists()
    assert config_path.exists()

    # 2. Training with config
    command_line = [
        "otx", "train",
        "--work_dir", str(tmp_dir_path),
        "--config", str(config_path),
    ]
    rc, stdout, _ = check_run(command_line)
    assert rc == 0
    assert "time elapsed" in str(stdout)
    assert checkpoint_path.exists()
    assert config_path.exists()

    # 3. Testing
    command_line = [
        "otx", "test",
        "--data.task", task,
        "--work_dir", str(tmp_dir_path),
        "--checkpoint", str(checkpoint_path),
        "--data.test_data_roots", TASK_CONFIGURATION[task]["test_data_roots"],
    ]
    rc, stdout, _ = check_run(command_line)
    assert rc == 0
    assert "time elapsed" in str(stdout)

    # 4. Prediction Single Image
    command_line = [
        "otx", "predict",
        "--work_dir", str(tmp_dir_path),
        "--checkpoint", str(checkpoint_path),
        "--img", TASK_CONFIGURATION[task]["sample"],
    ]
    rc, stdout, _ = check_run(command_line)
    assert rc == 0
    assert "time elapsed" in str(stdout)

    # 5. Export Openvino IR Model
    command_line = [
        "otx", "export",
        "--work_dir", str(tmp_dir_path),
        "--checkpoint", str(checkpoint_path),
    ]
    rc, stdout, _ = check_run(command_line)
    assert rc == 0
    assert "time elapsed" in str(stdout)
