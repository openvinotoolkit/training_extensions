# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import os
import subprocess
import yaml
from typing import List

from otx.api.entities.model_template import ModelTemplate, ModelCategory


def pytest_addoption(parser):
    """Add custom options for perf tests."""
    parser.addoption(
        "--model-type",
        action="store",
        default="all",
        help="Choose default|all. Defaults to all."
    )
    parser.addoption(
        "--data-size",
        action="store",
        default="all",
        help="Choose small|medium|large|all. Defaults to all."
    )
    parser.addoption(
        "--num-repeat",
        action="store",
        default=0,
        help="Overrides default per-data-size number of repeat setting. "
        "Random seeds are set to 0 ~ num_repeat-1 for the trials. "
        "Defaults to 0 (small=3, medium=3, large=1)."
    )
    parser.addoption(
        "--num-epoch",
        action="store",
        default=0,
        help="Overrides default per-model number of epoch setting. "
        "Defaults to 0 (per-model epoch & early-stopping)."
    )
    parser.addoption(
        "--eval-upto",
        action="store",
        default="all",
        help="Choose train|export|optimize. Defaults to train."
    )
    parser.addoption(
        "--data-root",
        action="store",
        default="data",
        help="Dataset root directory."
    )
    parser.addoption(
        "--output-dir",
        action="store",
        default="exp/perf",
        help="Output directory to save outputs."
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution."
    )


@pytest.fixture
def fxt_template(request: pytest.FixtureRequest):
    """Skip by model template."""
    model_type: str = request.config.getoption("--model-type")
    template: ModelTemplate = request.param
    if model_type == "default":
        if template.model_category == ModelCategory.OTHER:
            pytest.skip(f"{template.model_category} model")
    return template


@pytest.fixture
def fxt_benchmark_config(request: pytest.FixtureRequest):
    """Override benchmark config."""
    data_size_option: str = request.config.getoption("--data-size")
    data_size: str = request.param[0]
    datasets: List[str] = request.param[1]["datasets"]
    if data_size_option != "all":
        if data_size_option != data_size:
            pytest.skip(f"{data_size} datasets")

    num_epoch: int = request.param[1].get("num_epoch", 0)  # 0: per-model default
    num_epoch_override: int = request.config.getoption("--num-epoch")
    if num_epoch_override > 0:
        num_epoch = num_epoch_override

    num_repeat: int = request.param[1].get("num_repeat", 1)
    num_repeat_override: int = request.config.getoption("--num-repeat")
    if num_repeat_override > 0:
        num_repeat = num_repeat_override

    return data_size, datasets, num_epoch, num_repeat


@pytest.fixture
def fxt_commit_hash():
    """Short commit hash in short form."""
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


@pytest.fixture
def fxt_build_command(request: pytest.FixtureRequest, fxt_commit_hash: str, tmp_path_factory):
    """Research framework command builder."""
    eval_upto = request.config.getoption("--eval-upto")
    data_root = request.config.getoption("--data-root")
    data_root = os.path.abspath(data_root)
    output_dir = request.config.getoption("--output-dir")
    output_dir = os.path.abspath(output_dir + "-" + fxt_commit_hash)
    dry_run = request.config.getoption("--dry-run")

    def build_config(
        tag: str,
        model_template: ModelTemplate,
        datasets: List[str],
        num_epoch: int,
        num_repeat: int,
        track_resources: bool = False,
        params: str = "",
    ) -> dict:
        cfg = {}
        cfg["output_path"] = output_dir
        cfg["constants"] = {
            "dataroot": data_root,
        }
        cfg["variables"] = {
            "model": [model_template.model_template_id],
            "data": datasets,
        }
        cfg["repeat"] = num_repeat
        cfg["command"] = []
        if num_epoch > 0:
            params = params + f" --learning_pararmeters.num_iters {num_epoch}"
        resource_param = ""
        if track_resources:
            resource_param = " --track-resource-usage all"
        cfg["command"].append(
            "otx train ${model}"
            " --train-data-roots ${dataroot}/${data}"
            " --val-data-roots ${dataroot}/${data}"
            " --deterministic"
            f"{resource_param}"
            f" params {params}"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        if eval_upto == "train":
            return cfg

        cfg["command"].append(
            "otx export"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        if eval_upto == "export":
            return cfg

        cfg["command"].append(
            "otx optimize"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        return cfg

    def build_command(
        tag: str,
        model_template: ModelTemplate,
        datasets: List[str],
        num_epoch: int,
        num_repeat: int,
        track_resources: bool = False,
        params: str = "",
    ) -> List[str]:
        cfg = build_config(tag, model_template, datasets, num_epoch, num_repeat, track_resources, params)
        cfg_path = tmp_path_factory.mktemp("exp")/"cfg.yaml"
        print(cfg_path)
        with open(cfg_path, "w") as cfg_file:
            yaml.dump(cfg, cfg_file, indent=2,)
        cmd = [
            "python",
            "tools/experiment.py",
            "-f",
            cfg_path,
            "-d" if dry_run else "",
        ]
        return cmd

    return build_command


class OTXBenchmark:
    def __init__(self):
        pass
