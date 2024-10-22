# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
import torch
import yaml
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

from tests.utils import ExportCase2Test, run_main


@pytest.fixture(
    params=pytest.RECIPE_LIST,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def fxt_trained_model(
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
    request: pytest.FixtureRequest,
    tmp_path,
):
    recipe = request.param
    recipe_split = recipe.split("/")
    model_name = recipe_split[-1].split(".")[0]
    is_semisl = model_name.endswith("_semisl")
    task = recipe_split[-2] if not is_semisl else recipe_split[-3]

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

    if is_semisl:
        command_cfg.extend(
            [
                "--data.unlabeled_subset.data_root",
                fxt_target_dataset_per_task[f"{task}_semisl"],
            ],
        )

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    return recipe, task, model_name, tmp_path_train


def test_otx_e2e(
    fxt_trained_model,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
    fxt_export_list: list[ExportCase2Test],
    tmp_path: Path,
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
    recipe, task, model_name, tmp_path_train = fxt_trained_model
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
    ckpt_file = latest_dir / "best_checkpoint.ckpt"
    assert ckpt_file.exists()

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
        str(ckpt_file),
    ]
    # Zero-shot visual prompting needs to specify `infer_reference_info_root`
    if task in ["zero_shot_visual_prompting"]:
        idx_task = str(ckpt_file).split("/").index(f"otx_train_{model_name}")
        command_cfg.extend(
            [
                "--model.init_args.infer_reference_info_root",
                str(ckpt_file.parents[-idx_task] / f"otx_train_{model_name}/outputs/.latest/train"),
            ],
        )

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
            "dino_v2",
        ]
    ):
        return
    if task in ("visual_prompting", "zero_shot_visual_prompting"):
        fxt_export_list = [
            ExportCase2Test("ONNX", False, "exported_model_decoder.onnx"),
            ExportCase2Test("OPENVINO", False, "exported_model_decoder.xml"),
        ]  # TODO (sungchul): EXPORTABLE_CODE will be supported

    overrides = fxt_cli_override_command_per_task[task]

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for export_case in fxt_export_list:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--work_dir",
            str(tmp_path_test / "outputs" / export_case.export_format),
            *overrides,
            "--checkpoint",
            str(ckpt_file),
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
    if task in ("visual_prompting", "zero_shot_visual_prompting"):
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
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        fxt_accelerator,
        *overrides,
        "--checkpoint",
        exported_model_path,
    ]
    # Zero-shot visual prompting needs to specify `infer_reference_info_root`
    if task in ["zero_shot_visual_prompting"]:
        idx_task = str(ckpt_file).split("/").index(f"otx_train_{model_name}")
        command_cfg.extend(
            [
                "--model.init_args.infer_reference_info_root",
                str(ckpt_file.parents[-idx_task] / f"otx_train_{model_name}/outputs/.latest/train"),
            ],
        )

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

    if "dino" in model_name:
        return  # DINO is not supported.

    if "rtdetr" in model_name:
        return  # RT-DETR currently is not supported.

    tmp_path_test = tmp_path / f"otx_export_xai_{model_name}"
    for export_case in fxt_export_list:
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--work_dir",
            str(tmp_path_test / "outputs" / export_case.export_format),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_file),
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


def test_otx_explain_e2e(
    fxt_trained_model,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
    tmp_path: Path,
) -> None:
    """
    Test OTX CLI explain e2e command.

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """

    recipe, task, model_name, tmp_path_train = fxt_trained_model

    outputs_dir = tmp_path_train / "outputs"
    latest_dir = outputs_dir / ".latest"
    ckpt_file = latest_dir / "train" / "best_checkpoint.ckpt"
    assert ckpt_file.exists()

    if ("_cls" not in task) and (task not in ["detection", "instance_segmentation"]):
        pytest.skip("Supported only for classification, detection and instance segmentation task.")

    if "dino" in model_name:
        pytest.skip("DINO is not supported.")

    if "maskrcnn_r50_tv" in model_name:
        pytest.skip("MaskRCNN R50 Torchvision model doesn't support explain.")

    if "rtdetr" in recipe:
        pytest.skip("rtdetr model is not supported yet with explain.")

    # otx explain
    tmp_path_explain = tmp_path / f"otx_explain_{model_name}"
    command_cfg = [
        "otx",
        "explain",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_explain / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--seed",
        "0",
        "--dump",
        "True",
        "--checkpoint",
        str(ckpt_file),
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    outputs_dir = tmp_path_explain / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    assert (latest_dir / "saliency_map").exists()
    saliency_map = sorted((latest_dir / "saliency_map").glob(pattern="*.png"))
    sal_map = cv2.imread(str(saliency_map[0]))
    assert sal_map.shape[0] > 0
    assert sal_map.shape[1] > 0


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
        # TODO(Kirill):  Need to change this test when export enabled
        pytest.skip("OMZ doesn't have proper model for these types of tasks.")

    pytest.xfail("See ticket no. 135955")

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


def test_otx_hpo_e2e(
    fxt_trained_model,
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
    recipe, task, model_name, _ = fxt_trained_model

    if task.upper() == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
        pytest.skip("ZERO_SHOT_VISUAL_PROMPTING doesn't support HPO.")
    if "padim" in recipe:
        pytest.skip("padim model doesn't support HPO.")
    if model_name.endswith("_semisl"):
        pytest.skip("Semi-supervised learning model doesn't support HPO.")

    tmp_path_hpo = tmp_path / f"otx_hpo_{model_name}"
    tmp_path_hpo.mkdir(parents=True)

    command_cfg = [
        "otx",
        "train",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_hpo),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "1",
        "--run_hpo",
        "true",
        "--hpo_config.expected_time_ratio",
        "1",
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

    assert len([val for val in hpo_work_dor.rglob("*.json") if str(val.stem).isdigit()]) == 1


@pytest.mark.parametrize("task", pytest.TASK_LIST)
@pytest.mark.parametrize("bs_adapt_type", ["Safe", "Full"])
def test_otx_adaptive_bs_e2e(
    task: OTXTaskType,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
    fxt_xpu_support_task: list[OTXTaskType],
    bs_adapt_type: str,
) -> None:
    """
    Test adaptive batch size e2e commands with default template of each task.

    Args:
        task (OTXTaskType): The task to run adaptive batch size with.
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    if fxt_accelerator not in ["gpu", "xpu"]:
        pytest.skip("Adaptive batch size only supports GPU and XPU.")
    if fxt_accelerator == "xpu" and task not in fxt_xpu_support_task:
        pytest.skip(f"{task} doesn't support XPU.")
    if task not in DEFAULT_CONFIG_PER_TASK:
        pytest.skip(f"Task {task} is not supported in the auto-configuration.")
    if task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
        pytest.skip("ZERO_SHOT_VISUAL_PROMPTING doesn't support adaptive batch size.")

    task = task.lower()
    tmp_path_adap_bs = tmp_path / f"otx_adaptive_bs_{task}"
    tmp_path_adap_bs.mkdir(parents=True)

    command_cfg = [
        "otx",
        "train",
        "--task",
        task.upper(),
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_adap_bs),
        "--engine.device",
        fxt_accelerator,
        "--adaptive_bs",
        bs_adapt_type,
        "--max_epoch",
        "1",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)


@pytest.mark.parametrize("task", pytest.TASK_LIST)
def test_otx_configurable_input_size_e2e(
    task: OTXTaskType,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_open_subprocess: bool,
) -> None:
    """
    Test adaptive batch size e2e commands with default template of each task.

    Args:
        task (OTXTaskType): The task to run adaptive batch size with.
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    if task not in DEFAULT_CONFIG_PER_TASK:
        pytest.skip(f"Task {task} is not supported in the auto-configuration.")
    if task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
        pytest.skip(f"{task} doesn't support configurable input size.")
    if task == OTXTaskType.KEYPOINT_DETECTION:
        pytest.skip(f"{task} isn't prepared to run integration test.")

    task = task.lower()
    tmp_path_cfg_ipt_size = tmp_path / f"otx_configurable_input_size_{task}"
    tmp_path_cfg_ipt_size.mkdir(parents=True)

    command_cfg = [
        "otx",
        "train",
        "--task",
        task.upper(),
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_cfg_ipt_size),
        "--engine.device",
        fxt_accelerator,
        "--data.input_size",
        str(448),
        "--max_epoch",
        "1",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    best_ckpt_files = list(tmp_path_cfg_ipt_size.rglob("best_checkpoint.ckpt"))
    assert len(best_ckpt_files) != 0
    best_ckpt = torch.load(best_ckpt_files[0])
    assert best_ckpt["hyper_parameters"]["input_size"] == (448, 448)
    for param_name in best_ckpt["datamodule_hyper_parameters"]:
        if "subset" in param_name:
            assert best_ckpt["datamodule_hyper_parameters"][param_name].input_size == 448
