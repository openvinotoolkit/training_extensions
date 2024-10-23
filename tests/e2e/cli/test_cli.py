# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

from tests.e2e.cli.utils import run_main
from tests.utils import ExportCase2Test


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
    fxt_export_list: list[ExportCase2Test],
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
    recipe_split = recipe.split("/")
    model_name = recipe_split[-1].split(".")[0]
    is_semisl = model_name.endswith("_semisl")
    task = recipe_split[-2].upper() if not is_semisl else recipe_split[-3].upper()

    if task == OTXTaskType.INSTANCE_SEGMENTATION:
        is_tiling = "tile" in recipe
        dataset_path = fxt_target_dataset_per_task[task]["tiling" if is_tiling else "non_tiling"]
    else:
        dataset_path = fxt_target_dataset_per_task[task]

    if isinstance(dataset_path, dict) and "supervised" in dataset_path:
        dataset_path = dataset_path["supervised"]

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx",
        "train",
        "--config",
        recipe,
        "--data_root",
        str(dataset_path),
        "--work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "1" if task in (OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING) else "2",
        *fxt_cli_override_command_per_task[task],
    ]

    if is_semisl:
        command_cfg.extend(
            [
                "--data.unlabeled_subset.data_root",
                str(fxt_target_dataset_per_task[task]["unlabeled"]),
            ],
        )
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
        str(dataset_path),
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
    if task in (OTXTaskType.VISUAL_PROMPTING, OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING):
        fxt_export_list = [
            ExportCase2Test("ONNX", False, "exported_model_decoder.onnx"),
            ExportCase2Test("OPENVINO", False, "exported_model_decoder.xml"),
        ]
    elif "ANOMALY" in task or OTXTaskType.KEYPOINT_DETECTION in task:
        fxt_export_list = [
            ExportCase2Test("ONNX", False, "exported_model.onnx"),
            ExportCase2Test("OPENVINO", False, "exported_model.xml"),
        ]

    overrides = fxt_cli_override_command_per_task[task]
    if "anomaly" in task:
        overrides = {}  # Overrides are not needed in export

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for export_case in fxt_export_list:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            str(dataset_path),
            "--work_dir",
            str(tmp_path_test / "outputs" / export_case.export_format),
            *overrides,
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            export_case.export_format,
            "--export_demo_package",
            str(export_case.export_demo_package),
        ]

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        outputs_dir = tmp_path_test / "outputs" / export_case.export_format
        latest_dir = max(
            (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
            key=lambda p: p.stat().st_mtime,
        )
        assert latest_dir.exists()
        assert (latest_dir / export_case.expected_output).exists()

    # 4) infer of the exported models
    ov_output_dir = tmp_path_test / "outputs" / "OPENVINO"
    ov_files = list(ov_output_dir.rglob("exported*.xml"))
    if not ov_files:
        msg = "There is no OV IR."
        raise RuntimeError(msg)
    exported_model_path = str(ov_files[0])
    if task in (OTXTaskType.VISUAL_PROMPTING, OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING):
        recipe = str(Path(recipe).parents[0] / "openvino_model.yaml")

    overrides = fxt_cli_override_command_per_task[task]
    if "anomaly" in task:
        overrides = {}  # Overrides are not needed in infer

    command_cfg = [
        "otx",
        "test",
        "--config",
        recipe,
        "--data_root",
        str(dataset_path),
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
    if "instance_segmentation/rtmdet_inst_tiny" in recipe:
        return
    if ("_cls" not in task) and (task not in ["detection", "instance_segmentation", "semantic_segmentation"]):
        return  # Supported only for classification, detection and segmentation tasks.

    unsupported_models = ["dino", "rtdetr"]
    if any(model in model_name for model in unsupported_models):
        return  # The models are not supported.

    tmp_path_test = tmp_path / f"otx_export_xai_{model_name}"
    for export_case in fxt_export_list:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            str(dataset_path),
            "--work_dir",
            str(tmp_path_test / "outputs" / export_case.export_format),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            f"{export_case.export_format}",
            "--export_demo_package",
            str(export_case.export_demo_package),
            "--explain",
            "True",
        ]

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        fmt_dir = tmp_path_test / "outputs" / export_case.export_format
        assert fmt_dir.exists()
        fmt_latest_dir = max(
            (p for p in fmt_dir.iterdir() if p.is_dir() and p.name != ".latest"),
            key=lambda p: p.stat().st_mtime,
        )
        assert (fmt_latest_dir / f"{export_case.expected_output}").exists()


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
    import cv2

    recipe_split = recipe.split("/")
    model_name = recipe_split[-1].split(".")[0]
    is_semisl = model_name.endswith("_semisl")
    task = recipe_split[-2].upper() if not is_semisl else recipe_split[-3].upper()

    if is_semisl:
        pytest.skip("SEMI-SL is not supported for explain.")

    if task not in [
        OTXTaskType.MULTI_CLASS_CLS,
        OTXTaskType.MULTI_LABEL_CLS,
        OTXTaskType.H_LABEL_CLS,
        OTXTaskType.DETECTION,
        OTXTaskType.INSTANCE_SEGMENTATION,
    ]:
        pytest.skip("Supported only for classification, detection and instance segmentation task.")

    deterministic = "True"
    if task == OTXTaskType.INSTANCE_SEGMENTATION:
        # Determinism is not required for this test for instance_segmentation models.
        deterministic = "False"
        is_tiling = "tile" in recipe
        dataset_path = fxt_target_dataset_per_task[task]["tiling" if is_tiling else "non_tiling"]
    else:
        dataset_path = fxt_target_dataset_per_task[task]

    if isinstance(dataset_path, dict) and "supervised" in dataset_path:
        dataset_path = dataset_path["supervised"]

    if "dino" in model_name:
        pytest.skip("DINO is not supported.")

    # otx explain
    tmp_path_explain = tmp_path / f"otx_explain_{model_name}"
    command_cfg = [
        "otx",
        "explain",
        "--config",
        recipe,
        "--data_root",
        str(dataset_path),
        "--work_dir",
        str(tmp_path_explain / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--seed",
        "0",
        "--deterministic",
        deterministic,
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
    assert (latest_dir / "saliency_map").exists()
    saliency_maps = sorted((latest_dir / "saliency_map").glob(pattern="*.png"))
    sal_map = cv2.imread(str(saliency_maps[0]))
    assert sal_map.shape[0] > 0
    assert sal_map.shape[1] > 0

    sal_diff_thresh = 3
    reference_sal_vals = {
        # Classification
        "multi_label_cls_efficientnet_v2_light": (
            np.array([201, 209, 196, 158, 157, 119, 77], dtype=np.int16),
            "American_Crow_0031_25433_class_0_saliency_map.png",
        ),
        "h_label_cls_efficientnet_v2_light": (
            np.array([102, 141, 134, 79, 66, 92, 84], dtype=np.int16),
            "108_class_4_saliency_map.png",
        ),
        # Detection
        "detection_yolox_tiny": (
            np.array([182, 194, 187, 179, 188, 206, 215, 207, 177, 130], dtype=np.int16),
            "img_371_jpg_rf_a893e0bdc6fda0ba1b2a7f07d56cec23_class_0_saliency_map.png",
        ),
        "detection_ssd_mobilenetv2": (
            np.array([113, 139, 211, 190, 135, 91, 70, 103, 102, 89], dtype=np.int16),
            "img_371_jpg_rf_a893e0bdc6fda0ba1b2a7f07d56cec23_class_0_saliency_map.png",
        ),
        "detection_atss_mobilenetv2": (
            np.array([60, 95, 128, 107, 86, 111, 127, 125, 117, 116], dtype=np.int16),
            "img_371_jpg_rf_a893e0bdc6fda0ba1b2a7f07d56cec23_class_0_saliency_map.png",
        ),
        # Instance Segmentation
        "instance_segmentation_maskrcnn_efficientnetb2b": (
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
            "CDY_2018_class_0_saliency_map.png",
        ),
    }
    test_case_name = task + "_" + model_name
    if test_case_name in reference_sal_vals:
        actual_sal_vals = cv2.imread(str(latest_dir / "saliency_map" / reference_sal_vals[test_case_name][1]))
        if test_case_name == "instance_segmentation_maskrcnn_efficientnetb2b":
            # Take lower corner values due to map sparsity of InstSeg
            actual_sal_vals = (actual_sal_vals[-10:, -1, 0]).astype(np.int16)
        else:
            actual_sal_vals = (actual_sal_vals[:10, 0, 0]).astype(np.int16)
        ref_sal_vals = reference_sal_vals[test_case_name][0]
        assert np.max(np.abs(actual_sal_vals - ref_sal_vals) <= sal_diff_thresh)


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
    task = task.upper()
    if task not in DEFAULT_CONFIG_PER_TASK:
        pytest.skip(f"Task {task} is not supported in the auto-configuration.")
    if task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
        pytest.skip("ZERO_SHOT_VISUAL_PROMPTING doesn't support HPO.")

    # Need to change model to stfpm because default anomaly model is 'padim' which doesn't support HPO
    model_cfg = []
    if task in {
        OTXTaskType.ANOMALY_CLASSIFICATION,
        OTXTaskType.ANOMALY_DETECTION,
        OTXTaskType.ANOMALY_SEGMENTATION,
    }:
        model_cfg = ["--config", str(DEFAULT_CONFIG_PER_TASK[task].parent / "stfpm.yaml")]

    if task == OTXTaskType.INSTANCE_SEGMENTATION:
        dataset_path = fxt_target_dataset_per_task[task]["non_tiling"]
    else:
        dataset_path = fxt_target_dataset_per_task[task]

    if isinstance(dataset_path, dict) and "supervised" in dataset_path:
        dataset_path = dataset_path["supervised"]

    tmp_path_hpo = tmp_path / f"otx_hpo_{task.lower()}"
    tmp_path_hpo.mkdir(parents=True)

    command_cfg = [
        "otx",
        "train",
        *model_cfg,
        "--task",
        task,
        "--data_root",
        str(dataset_path),
        "--work_dir",
        str(tmp_path_hpo),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "1" if task in (OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING) else "2",
        "--run_hpo",
        "true",
        "--hpo_config.expected_time_ratio",
        "2",
        "--hpo_config.num_workers",
        "1",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

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
