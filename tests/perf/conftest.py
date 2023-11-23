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
        help="Overrides default per-data-size settings. Defaults to 0, which means no override."
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
def fxt_data_setting(request: pytest.FixtureRequest):
    """Skip by dataset size."""
    data_size_option: str = request.config.getoption("--data-size")
    data_size: str = request.param[0]
    datasets: List[str] = request.param[1]["datasets"]
    num_repeat: int = request.param[1]["num_repeat"]
    num_repeat_override: int = request.config.getoption("--num-repeat")
    if num_repeat_override > 0:
        num_repeat = num_repeat_override

    if data_size_option != "all":
        if data_size_option != data_size:
            pytest.skip(f"{data_size} datasets")
    return data_size, datasets, num_repeat


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

    def build_config(
        tag: str,
        model_template: ModelTemplate,
        datasets: List[str],
        num_repeat: int,
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
        cfg["command"].append(
            "otx train ${model}"
            " --train-data-roots ${dataroot}/${data}"
            " --val-data-roots ${dataroot}/${data}"
            " --track-resource-usage all"
            " --deterministic"
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
        num_repeat: int,
        params: str = "",
    ) -> List[str]:
        cfg = build_config(tag, model_template, datasets, num_repeat, params)
        cfg_path = tmp_path_factory.mktemp("exp")/"cfg.yaml"
        print(cfg_path)
        with open(cfg_path, "w") as cfg_file:
            yaml.dump(cfg, cfg_file, indent=2,)
        cmd = [
            "python",
            "tools/experiment.py",
            "-d",
            "-f",
            cfg_path,
        ]
        return cmd

    return build_command


class OTXBenchmark:
    def __init__(self):
        pass
