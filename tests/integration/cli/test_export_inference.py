# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import inspect
import logging
from pathlib import Path

import pandas as pd
import pytest

from tests.integration.cli.utils import run_main

log = logging.getLogger(__name__)

# This assumes have OTX installed in environment.
otx_module = importlib.import_module("otx")
RECIPE_PATH = Path(inspect.getfile(otx_module)).parent / "recipe"
RECIPE_LIST = [str(p) for p in RECIPE_PATH.glob("**/*.yaml") if "_base_" not in p.parts]
RECIPE_OV_LIST = [str(p) for p in RECIPE_PATH.glob("**/openvino_model.yaml") if "_base_" not in p.parts]
RECIPE_LIST = set(RECIPE_LIST) - set(RECIPE_OV_LIST)


def _check_relative_metric_diff(ref: float, value: float, eps: float) -> None:
    assert ref >= 0
    assert value >= 0
    assert eps >= 0

    avg = max(0.5 * (ref + value), 1e-9)
    diff = abs(value - ref)

    assert diff / avg <= eps, f"Relative difference exceeded {eps} threshold. Absolute difference: {diff}"


@pytest.fixture(scope="module", autouse=True)
def fxt_local_seed() -> int:
    """The number of repetition for each test case.

    The random seed will be set for [0, fxt_num_repeat - 1]. Default is one.
    """
    selected_seed = 7
    msg = f"seed : {selected_seed}"
    log.info(msg)
    return selected_seed


TASK_NAME_TO_MAIN_METRIC_NAME = {
    "semantic_segmentation": "test/Dice",
    "multi_label_cls": "test/accuracy",
    "multi_class_cls": "test/accuracy",
    "detection": "test/map_50",
    "instance_segmentation": "test/map_50",
}


@pytest.mark.parametrize("recipe", RECIPE_LIST, ids=lambda x: "/".join(Path(x).parts[-2:]))
def test_otx_export_infer(
    recipe: str,
    tmp_path: Path,
    fxt_local_seed: int,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_accelerator: str,
    fxt_open_subprocess: bool,
    request: pytest.FixtureRequest,
) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx train' with 2 epochs training
    - 'otx test' with output checkpoint from 'otx train'
    - 'otx export' with output checkpoint from 'otx train'
    - 'otx test' with the exported to ONNX/IR model model
    - compare accuracy of the exported model vs the original accuracy

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]

    if task not in TASK_NAME_TO_MAIN_METRIC_NAME:
        pytest.skip(f"Inference pipeline for {recipe} is not implemented")
    elif (task == "detection" and "atss_mobilenetv2" not in recipe) or (
        task == "instance_segmentation" and "maskrcnn_efficientnetb2b" not in recipe
    ):
        pytest.skip("To prevent memory bug from aborting integration test, test single model per task.")
    elif "tile" in recipe:
        pytest.skip("Exporting tiling model isn't suppored yet.")
    elif "otx_dino_v2_linear_probe" in recipe:
        pytest.skip("Test pipeline is different between torch and ov model.")  # NOTE Enable test after making them same

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
        "--engine.work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        "--seed",
        f"{fxt_local_seed}",
        "--deterministic",
        "warn",
        *fxt_cli_override_command_per_task[task],
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

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
        str(tmp_path_test / "outputs" / "torch"),
        "--engine.device",
        fxt_accelerator,
        *fxt_cli_override_command_per_task[task],
        "--checkpoint",
        str(ckpt_files[-1]),
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    # 3) otx export
    format_to_ext = {"OPENVINO": "xml"}  # [TODO](@Vlad): extend to "ONNX": "onnx"

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for fmt in format_to_ext:
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
        assert (tmp_path_test / "outputs" / f"exported_model.{format_to_ext[fmt]}").exists()

    # 4) infer of the exported models
    task = recipe.split("/")[-2]
    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    if "_cls" in recipe:
        export_test_recipe = f"src/otx/recipe/classification/{task}/openvino_model.yaml"
    else:
        export_test_recipe = f"src/otx/recipe/{task}/openvino_model.yaml"
    exported_model_path = str(tmp_path_test / "outputs" / "exported_model.xml")

    command_cfg = [
        "otx",
        "test",
        "--config",
        export_test_recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs" / "openvino"),
        "--engine.device",
        "cpu",
        *fxt_cli_override_command_per_task[task],
        "--model.model_name",
        exported_model_path,
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    assert (tmp_path_test / "outputs").exists()

    # 5) test optimize
    command_cfg = [
        "otx",
        "optimize",
        "--config",
        export_test_recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs"),
        "--engine.device",
        "cpu",
        *fxt_cli_override_command_per_task[task],
        "--model.model_name",
        exported_model_path,
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    assert (tmp_path_test / "outputs").exists()
    exported_model_path = str(tmp_path_test / "outputs" / "optimized_model.xml")

    # 6) test optimized model
    command_cfg = [
        "otx",
        "test",
        "--config",
        export_test_recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--engine.work_dir",
        str(tmp_path_test / "outputs" / "nncf_ptq"),
        "--engine.device",
        "cpu",
        *fxt_cli_override_command_per_task[task],
        "--model.model_name",
        exported_model_path,
    ]

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    assert (tmp_path_test / "outputs").exists()

    df_torch = pd.read_csv(next((tmp_path_test / "outputs" / "torch").glob("**/metrics.csv")))
    df_openvino = pd.read_csv(next((tmp_path_test / "outputs" / "openvino").glob("**/metrics.csv")))
    df_nncf_ptq = pd.read_csv(next((tmp_path_test / "outputs" / "nncf_ptq").glob("**/metrics.csv")))

    metric_name = TASK_NAME_TO_MAIN_METRIC_NAME[task]

    assert metric_name in df_torch.columns
    assert metric_name in df_openvino.columns
    assert metric_name in df_nncf_ptq.columns

    torch_acc = df_torch[metric_name].item()
    ov_acc = df_openvino[metric_name].item()
    ptq_acc = df_nncf_ptq[metric_name].item()  # noqa: F841

    msg = f"Recipe: {recipe}, (torch_accuracy, ov_accuracy): {torch_acc} , {ov_acc}"
    log.info(msg)

    # Not compare w/ instance segmentation because training isn't able to be deterministic, which can lead to unstable test result.
    if "maskrcnn_efficientnetb2b" in recipe:
        return

    if "multi_label_cls/mobilenet_v3_large_light" in request.node.name:
        msg = "multi_label_cls/mobilenet_v3_large_light exceeds the following threshold = 0.1"
        pytest.xfail(msg)

    _check_relative_metric_diff(torch_acc, ov_acc, 0.1)
