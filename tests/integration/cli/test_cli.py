# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import numpy as np
import pytest
import yaml
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

from tests.integration.cli.utils import run_main


@pytest.mark.parametrize(
    "recipe",
    pytest.RECIPE_LIST,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def test_otx_e2e(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx train' with 2 epochs training
    - 'otx test' with output checkpoint from 'otx train'
    - 'otx export' with output checkpoint from 'otx train'
    - 'otx test' with the exported to ONNX/IR model

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]
    if task in ("action_classification"):
        pytest.xfail(reason="xFail until this root cause is resolved on the Datumaro side.")

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx",
        "train",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    if task in ("zero_shot_visual_prompting"):
        pytest.skip("Full CLI test is not applicable to this task.")

    outputs_dir = tmp_path_train / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    # Currently, a simple output check
    assert latest_dir.exists()
    assert (latest_dir / "configs.yaml").exists()
    # Check Configs file
    with (latest_dir / "configs.yaml").open() as file:
        train_output_config = yaml.safe_load(file)
    assert "model" in train_output_config
    assert "data" in train_output_config
    assert "engine" in train_output_config
    assert (latest_dir / "csv").exists()
    assert (latest_dir / "checkpoints").exists()
    ckpt_files = list((latest_dir / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0

    # 2) otx test
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        fxt_accelerator,
        *fxt_cli_override_command_per_task[task],
        "--checkpoint",
        str(ckpt_files[-1]),
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    outputs_dir = tmp_path_test / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    assert latest_dir.exists()
    assert (latest_dir / "csv").exists()

    # 3) otx export
    if any(
        task_name in recipe
        for task_name in [
            "h_label_cls",
            "detection",
            "dino_v2",
            "instance_segmentation",
            "action",
            "visual_prompting",
        ]
    ):
        return

    format_to_file = {
        "ONNX": "exported_model.onnx",
        "OPENVINO": "exported_model.xml",
        "EXPORTABLE_CODE": "exportable_code.zip",
    }

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for fmt in format_to_file:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--work_dir",
            str(tmp_path_test / "outputs" / fmt),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            f"{fmt}",
        ]

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        outputs_dir = tmp_path_test / "outputs" / fmt
        latest_dir = max(
            (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
            key=lambda p: p.stat().st_mtime,
        )
        assert latest_dir.exists()
        assert (latest_dir / f"{format_to_file[fmt]}").exists()

    # 4) infer of the exported models
    ov_output_dir = tmp_path_test / "outputs" / "OPENVINO"
    ov_latest_dir = max(
        (p for p in ov_output_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    exported_model_path = str(ov_latest_dir / "exported_model.xml")

    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        *fxt_cli_override_command_per_task[task],
        "--checkpoint",
        exported_model_path,
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    outputs_dir = tmp_path_test / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    assert latest_dir.exists()

    # 5) otx export with XAI
    if "_cls" not in task or "dino" in model_name:
        return

    format_to_file = {
        "ONNX": "exported_model.onnx",
        "OPENVINO": "exported_model.xml",
        "EXPORTABLE_CODE": "exportable_code.zip",
    }

    tmp_path_test = tmp_path / f"otx_export_xai_{model_name}"
    for fmt in format_to_file:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--work_dir",
            str(tmp_path_test / "outputs" / fmt),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            f"{fmt}",
            "--explain",
            "True",
        ]

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        fmt_dir = tmp_path_test / "outputs" / fmt
        assert fmt_dir.exists()
        fmt_latest_dir = max(
            (p for p in fmt_dir.iterdir() if p.is_dir() and p.name != ".latest"),
            key=lambda p: p.stat().st_mtime,
        )
        assert (fmt_latest_dir / f"{format_to_file[fmt]}").exists()

    # 5) otx export with XAI
    if ("_cls" not in task) and (task != "detection"):
        pytest.skip("Supported only for classification and detection task.")
    
    if "dino" in model_name or "deit" in model_name:
        pytest.skip("Dino is not supported.")

    format_to_file = {
        "ONNX": "exported_model.onnx",
        "OPENVINO": "exported_model.xml",
        "EXPORTABLE_CODE": "exportable_code.zip",
    }

    tmp_path_test = tmp_path / f"otx_export_xai_{model_name}"
    for fmt in format_to_file:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--engine.work_dir",
            str(tmp_path_test / "outputs"),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            f"{fmt}",
            "--explain",
            "True",
        ]

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        assert (tmp_path_test / "outputs").exists()
        assert (tmp_path_test / "outputs" / f"{format_to_file[fmt]}").exists()


@pytest.mark.parametrize(
    "recipe",
    pytest.RECIPE_LIST,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def test_otx_explain_e2e(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
) -> None:
    """
    Test OTX CLI explain e2e command.

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    if "tile" in recipe:
        pytest.skip("Explain is not supported for tiling yet.")

    import cv2

    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if "_cls" not in task:
        pytest.skip("Supported only for classification.")

    if "dino" in model_name:
        pytest.skip("DINO is not supported.")

    # otx explain
    tmp_path_explain = tmp_path / f"otx_explain_{model_name}"
    command_cfg = [
        "otx",
        "explain",
        "--config",
        recipe,
        "--model.num_classes",
        "1000",
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_explain / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--seed",
        "0",
        "--deterministic",
        "True",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    outputs_dir = tmp_path_explain / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    assert latest_dir.exists()
    assert (latest_dir / "saliency_map.tiff").exists()
    sal_map = cv2.imread(str(latest_dir / "saliency_map.tiff"))
    assert sal_map.shape[0] > 0
    assert sal_map.shape[1] > 0

    # TMP: remove reference_sal_vals


# @pytest.mark.skipif(len(pytest.RECIPE_OV_LIST) < 1, reason="No OV recipe found.")
@pytest.mark.parametrize(
    "ov_recipe",
    pytest.RECIPE_OV_LIST,
)
def test_otx_ov_test(
    ov_recipe: str,
    tmp_path: Path,
    fxt_target_dataset_per_task: dict,
    fxt_open_subprocess: bool,
) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx test' with OV model

    Args:
        recipe (str): The OV recipe to use for testing. (eg. 'classification/openvino_model.yaml')
        tmp_path (Path): The temporary path for storing the testing outputs.

    Returns:
        None
    """
    task = ov_recipe.split("/")[-2]
    model_name = ov_recipe.split("/")[-1].split(".")[0]

    if task in ["multi_label_cls", "instance_segmentation", "h_label_cls"]:
        # OMZ doesn't have proper model for Pytorch MaskRCNN interface
        # TODO(Kirill):  Need to change this test when export enabled #noqa: TD003
        pytest.skip("OMZ doesn't have proper model for these types of tasks.")

    if task in ["action_classification"]:
        pytest.skip("Action classification test will be enabled after solving Datumaro issue.")

    # otx test
    tmp_path_test = tmp_path / f"otx_test_{task}_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        ov_recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        "--disable-infer-num-classes",
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    outputs_dir = tmp_path_test / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    assert latest_dir.exists()
    assert (latest_dir / "csv").exists()
    metric_result = list((latest_dir / "csv").glob(pattern="**/metrics.csv"))
    assert len(metric_result) > 0


@pytest.mark.parametrize("task", pytest.TASK_LIST)
def test_otx_hpo_e2e(
    task: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
) -> None:
    """
    Test HPO e2e commands with default template of each task.

    Args:
        task (OTXTaskType): The task to run HPO with.
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    if task in ("action_classification"):
        pytest.xfail(reason="xFail until this root cause is resolved on the Datumaro side.")
    if task not in DEFAULT_CONFIG_PER_TASK:
        pytest.skip(f"Task {task} is not supported in the auto-configuration.")

    task = task.lower()
    tmp_path_hpo = tmp_path / f"otx_hpo_{task}"
    tmp_path_hpo.mkdir(parents=True)

    command_cfg = [
        "otx",
        "train",
        "--task",
        task.upper(),
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_hpo),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        "--run_hpo",
        "true",
        "--hpo_config.expected_time_ratio",
        "2",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    # zero_shot_visual_prompting doesn't support HPO. Check just there is no error.
    if task in ("zero_shot_visual_prompting"):
        return

    latest_dir = max(
        (p for p in tmp_path_hpo.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    hpo_work_dor = latest_dir / "hpo"
    assert hpo_work_dor.exists()
    assert len([val for val in hpo_work_dor.rglob("*.json") if str(val.stem).isdigit()]) == 2
