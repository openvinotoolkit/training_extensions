# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import os
import subprocess
import yaml
from pathlib import Path
from typing import List
from datetime import datetime

from otx.api.entities.model_template import ModelTemplate, ModelCategory


def pytest_addoption(parser):
    """Add custom options for perf tests."""
    parser.addoption(
        "--model-type",
        action="store",
        default="all",
        choices=("default", "all"),
        help="Choose default|all. Defaults to all.",
    )
    parser.addoption(
        "--data-size",
        action="store",
        default="all",
        choices=("small", "medium", "large", "all"),
        help="Choose small|medium|large|all. Defaults to all.",
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
        choices=("train", "export", "optimize"),
        help="Choose train|export|optimize. Defaults to train."
    )
    parser.addoption(
        "--data-root",
        action="store",
        default="data",
        help="Dataset root directory."
    )
    parser.addoption(
        "--output-root",
        action="store",
        default="exp/perf",
        help="Output root directory."
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution."
    )


@pytest.fixture
def fxt_commit_hash():
    """Short commit hash."""
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


@pytest.fixture
def fxt_model_id(request: pytest.FixtureRequest):
    """Skip by model category."""
    model_type: str = request.config.getoption("--model-type")
    model_template: ModelTemplate = request.param
    if model_type == "default":
        if model_template.model_category == ModelCategory.OTHER:
            pytest.skip(f"{model_template.model_category} category model")
    return model_template.model_template_id


@pytest.fixture
def fxt_benchmark(request: pytest.FixtureRequest, fxt_commit_hash: str):
    """Configure benchmark."""
    # Skip by dataset size
    data_size_option: str = request.config.getoption("--data-size")
    data_size: str = request.param[0]
    if data_size_option != "all":
        if data_size_option != data_size:
            pytest.skip(f"{data_size} datasets")

    # Options
    cfg: dict = request.param[1].copy()

    num_epoch_override: int = int(request.config.getoption("--num-epoch"))
    if num_epoch_override > 0:  # 0: use default
        cfg["num_epoch"] = num_epoch_override
    if "test_speed" in request.node.name:
        if cfg.get("num_epoch", 0) == 0:  # No user options
            cfg["num_epoch"] = 2

    num_repeat_override: int = int(request.config.getoption("--num-repeat"))
    if num_repeat_override > 0:  # 0: use default
        cfg["num_repeat"] = num_repeat_override

    cfg["eval_upto"] = request.config.getoption("--eval-upto")
    cfg["data_root"] = request.config.getoption("--data-root")
    output_root = request.config.getoption("--output-root")
    output_dir = fxt_commit_hash + "-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["output_root"] = str(Path(output_root) / output_dir)
    cfg["dry_run"] = request.config.getoption("--dry-run")

    tags = cfg.get("tags", {})
    tags["data_size"] = data_size
    cfg["tags"] = tags

    # Create benchmark
    benchmark = OTXBenchmark(
        **cfg,
    )

    return benchmark


class OTXBenchmark:
    def __init__(
        self,
        datasets: List[str],
        data_root: str = "data",
        num_epoch: int = 0,
        num_repeat: int = 0,
        train_params: dict = {},
        track_resources: bool = False,
        eval_upto: str = "train",
        output_root: str = "otx-benchmark",
        dry_run: bool = False,
        tags: dict = {},
    ):
        self.datasets = datasets
        self.data_root = data_root
        self.num_epoch = num_epoch
        self.num_repeat = num_repeat
        self.train_params = train_params
        self.track_resources = track_resources
        self.eval_upto = eval_upto
        self.output_root = output_root
        self.dry_run = dry_run
        self.tags = tags

    def build_command(
        self,
        model_id: str,
        train_params: dict = {},
        tags: dict = {},
    ) -> List[str]:
        cfg = self._build_config(model_id, train_params, tags)
        cfg_dir = Path(self.output_root)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "cfg.yaml"
        print(cfg_path)
        with open(cfg_path, "w") as cfg_file:
            yaml.dump(cfg, cfg_file, indent=2,)
        cmd = [
            "python",
            "tools/experiment.py",
            "-f",
            cfg_path,
        ]
        if self.dry_run:
            cmd.append("-d")
        return cmd

    def _build_config(
        self,
        model_id: str,
        train_params: dict = {},
        tags: dict = {},
    ) -> dict:
        all_train_params = self.train_params.copy()
        all_train_params.update(train_params)
        all_tags = self.tags.copy()
        all_tags.update(tags)

        cfg = {}
        cfg["tags"] = all_tags  # metadata
        cfg["output_path"] = os.path.abspath(self.output_root)
        cfg["constants"] = {
            "dataroot": os.path.abspath(self.data_root),
        }
        cfg["variables"] = {
            "model": [model_id],
            "data": self.datasets,
            **{k: [v] for k, v in all_tags.items()},  # To be shown in result file
        }
        cfg["repeat"] = self.num_repeat
        cfg["command"] = []
        resource_param = ""
        if self.track_resources:
            resource_param = "--track-resource-usage all"
        if self.num_epoch > 0:
            all_train_params["learning_parameters.num_iters"] = self.num_epoch
        params_str = " ".join([f"--{k} {v}" for k, v in all_train_params.items()])
        cfg["command"].append(
            "otx train ${model}"
            " --train-data-roots ${dataroot}/${data}"
            " --val-data-roots ${dataroot}/${data}"
            " --deterministic"
            f" {resource_param}"
            f" params {params_str}"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        if self.eval_upto == "train":
            return cfg

        cfg["command"].append(
            "otx export"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        if self.eval_upto == "export":
            return cfg

        cfg["command"].append(
            "otx optimize"
        )
        cfg["command"].append(
            "otx eval"
            " --test-data-roots ${dataroot}/${data}"
        )
        return cfg
