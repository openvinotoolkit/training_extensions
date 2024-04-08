# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import numpy as np
import pytest
import yaml
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

from tests.e2e.cli.utils import run_main


@pytest.mark.parametrize(
    "recipe",
    pytest.RECIPE_LIST,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def test_otx_e2e_cli(
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
        "1" if task in ("zero_shot_visual_prompting") else "2",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

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
        ]
    ):
        return

    if task in ("visual_prompting", "zero_shot_visual_prompting"):
        format_to_file = {
            "ONNX": "exported_model_decoder.onnx",
            "OPENVINO": "exported_model_decoder.xml",
            # TODO (sungchul): EXPORTABLE_CODE will be supported # noqa: TD003
        }
    else:
        format_to_file = {
            "ONNX": "exported_model.onnx",
            "OPENVINO": "exported_model.xml",
            "EXPORTABLE_CODE": "exportable_code.zip",
        }

    overrides = fxt_cli_override_command_per_task[task]
    if "anomaly" in task:
        overrides = {}  # Overrides are not needed in export

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
            *overrides,
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

    overrides = fxt_cli_override_command_per_task[task]
    if "anomaly" in task:
        overrides = {}  # Overrides are not needed in infer

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
        *overrides,
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
    if ("_cls" not in task) and (task not in ["detection", "instance_segmentation"]):
        return  # Supported only for classification, detection and instance segmentation task.

    if "dino" in model_name or "rtmdet_inst_tiny" in model_name:
        return  # DINO and Rtmdet_tiny are not supported.

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


@pytest.mark.parametrize(
    "recipe",
    pytest.RECIPE_LIST,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def test_otx_explain_e2e_cli(
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

    if ("_cls" not in task) and (task not in ["detection", "instance_segmentation"]):
        pytest.skip("Supported only for classification, detection and instance segmentation task.")

    if "dino" in model_name or "rtmdet_inst_tiny" in model_name:
        pytest.skip("DINO and Rtmdet_tiny are not supported.")

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
        "--dump",
        "True",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    outputs_dir = tmp_path_explain / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    assert (latest_dir / "saliency_maps").exists()
    saliency_maps = sorted((latest_dir / "saliency_maps").glob(pattern="*.png"))
    sal_map = cv2.imread(str(saliency_maps[0]))
    assert sal_map.shape[0] > 0
    assert sal_map.shape[1] > 0

    sal_diff_thresh = 3
    reference_sal_vals = {
        # Classification
        "multi_label_cls_efficientnet_v2_light": (
            np.array([201, 209, 196, 158, 157, 119, 77], dtype=np.uint8),
            "American_Crow_0031_25433_class_0_saliency_map.png",
        ),
        "h_label_cls_efficientnet_v2_light": (
            np.array([102, 141, 134, 79, 66, 92, 84], dtype=np.uint8),
            "108_class_4_saliency_map.png",
        ),
        # Detection
        "detection_yolox_tiny": (
            np.array([182, 194, 187, 179, 188, 206, 215, 207, 177, 130], dtype=np.uint8),
            "img_371_jpg_rf_a893e0bdc6fda0ba1b2a7f07d56cec23_class_0_saliency_map.png",
        ),
        "detection_ssd_mobilenetv2": (
            np.array([118, 188, 241, 213, 160, 120, 86, 94, 111, 138], dtype=np.uint8),
            "img_371_jpg_rf_a893e0bdc6fda0ba1b2a7f07d56cec23_class_0_saliency_map.png",
        ),
        "detection_atss_mobilenetv2": (
            np.array([29, 39, 55, 69, 80, 88, 92, 86, 100, 88], dtype=np.uint8),
            "img_371_jpg_rf_a893e0bdc6fda0ba1b2a7f07d56cec23_class_0_saliency_map.png",
        ),
        # Instance Segmentation
        "instance_segmentation_maskrcnn_efficientnetb2b": (
            np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.uint8),
            "CDY_2018_class_0_saliency_map.png",
        ),
    }
    test_case_name = task + "_" + model_name
    if test_case_name in reference_sal_vals:
        actual_sal_vals = cv2.imread(str(latest_dir / "saliency_maps" / reference_sal_vals[test_case_name][1]))
        if test_case_name == "instance_segmentation_maskrcnn_efficientnetb2b":
            # Take lower corner values due to map sparsity of InstSeg
            actual_sal_vals = (actual_sal_vals[-10:, -1, 0]).astype(np.uint16)
        else:
            actual_sal_vals = (actual_sal_vals[:10, 0, 0]).astype(np.uint16)
        ref_sal_vals = reference_sal_vals[test_case_name][0]
        assert np.max(np.abs(actual_sal_vals - ref_sal_vals) <= sal_diff_thresh)


# @pytest.mark.skipif(len(pytest.RECIPE_OV_LIST) < 1, reason="No OV recipe found.")
@pytest.mark.parametrize(
    "ov_recipe",
    pytest.RECIPE_OV_LIST,
)
def test_otx_ov_test_cli(
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

    if task in [
        "multi_label_cls",
        "instance_segmentation",
        "h_label_cls",
        "visual_prompting",
        "zero_shot_visual_prompting",
        "anomaly_classification",
        "anomaly_detection",
        "anomaly_segmentation",
        "action_classification",
    ]:
        # OMZ doesn't have proper model for Pytorch MaskRCNN interface
        # TODO(Kirill):  Need to change this test when export enabled #noqa: TD003
        pytest.skip("OMZ doesn't have proper model for these types of tasks.")

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
def test_otx_hpo_e2e_cli(
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
        "1" if task in ("zero_shot_visual_prompting") else "2",
        "--run_hpo",
        "true",
        "--hpo_config.expected_time_ratio",
        "2",
        "--hpo_config.num_workers",
        "1",
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
    # Anomaly doesn't do validation. Check just there is no error.
    if task.startswith("anomaly"):
        return

    assert len([val for val in hpo_work_dor.rglob("*.json") if str(val.stem).isdigit()]) == 2
