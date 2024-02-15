# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import inspect
from pathlib import Path

import pytest
import yaml

from tests.integration.cli.utils import run_main

# This assumes have OTX installed in environment.
otx_module = importlib.import_module("otx")
RECIPE_PATH = Path(inspect.getfile(otx_module)).parent / "recipe"
RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("**/*.yaml") if "_base_" not in p.parts]
RECIPE_OV_LIST = [str(p) for p in RECIPE_PATH.glob("**/openvino_model.yaml") if "_base_" not in p.parts]
RECIPE_LIST = set(RECIPE_LIST) - set(RECIPE_OV_LIST)


@pytest.mark.parametrize("recipe", RECIPE_LIST)
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
    tile_param = fxt_cli_override_command_per_task["tile"] if "tile" in recipe else []
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
        "--engine.work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        *fxt_cli_override_command_per_task[task],
        *tile_param,
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    if task in ("zero_shot_visual_prompting"):
        pytest.skip("Full CLI test is not applicable to this task.")

    # Currently, a simple output check
    assert (tmp_path_train / "outputs").exists()
    assert (tmp_path_train / "outputs" / "configs.yaml").exists()
    # Check Configs file
    with (tmp_path_train / "outputs" / "configs.yaml").open() as file:
        train_output_config = yaml.safe_load(file)
    assert "model" in train_output_config
    assert "data" in train_output_config
    assert "engine" in train_output_config
    assert (tmp_path_train / "outputs" / "csv").exists()
    assert (tmp_path_train / "outputs" / "checkpoints").exists()
    ckpt_files = list((tmp_path_train / "outputs" / "checkpoints").glob(pattern="epoch_*.ckpt"))
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
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        fxt_accelerator,
        *fxt_cli_override_command_per_task[task],
        "--checkpoint",
        str(ckpt_files[-1]),
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "csv").exists()

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
            "--engine.work_dir",
            str(tmp_path_test / "outputs"),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            f"{fmt}",
        ]

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        assert (tmp_path_test / "outputs").exists()
        assert (tmp_path_test / "outputs" / f"{format_to_file[fmt]}").exists()

    # 4) infer of the exported models
    exported_model_path = str(tmp_path_test / "outputs" / "exported_model.xml")

    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        *fxt_cli_override_command_per_task[task],
        "--checkpoint",
        exported_model_path,
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    assert (tmp_path_test / "outputs").exists()

    # 5) otx export with XAI
    if "_cls" not in task or "dino" in model_name or "deit" in model_name:
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


@pytest.mark.parametrize("recipe", RECIPE_LIST)
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
    import cv2

    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if ("_cls" not in task) and (task != "detection"):
        pytest.skip("Supported only for classification and detection task.")

    if "dino" in model_name:
        pytest.skip("Dino is not supported.")

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
        "--engine.work_dir",
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

    assert (tmp_path_explain / "outputs").exists()
    assert (tmp_path_explain / "outputs" / "saliency_map.tiff").exists()
    sal_map = cv2.imread(str(tmp_path_explain / "outputs" / "saliency_map.tiff"))
    assert sal_map.shape[0] > 0
    assert sal_map.shape[1] > 0

    # TMP: remove reference_sal_vals


@pytest.mark.parametrize("recipe", RECIPE_OV_LIST)
def test_otx_ov_test(
    recipe: str,
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
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if task in ["multi_label_cls", "instance_segmentation", "h_label_cls"]:
        # OMZ doesn't have proper model for Pytorch MaskRCNN interface
        # TODO(Kirill):  Need to change this test when export enabled #noqa: TD003
        pytest.skip("OMZ doesn't have proper model for these types of tasks.")

    # otx test
    tmp_path_test = tmp_path / f"otx_test_{task}_{model_name}"
    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        "--disable-infer-num-classes",
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "csv").exists()
    metric_result = list((tmp_path_test / "outputs" / "csv").glob(pattern="**/metrics.csv"))
    assert len(metric_result) > 0
