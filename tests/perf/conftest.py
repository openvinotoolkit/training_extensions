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
from .benchmark import OTXBenchmark


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
        "Defaults to 0 (small=3, medium=3, large=1).",
    )
    parser.addoption(
        "--num-epoch",
        action="store",
        default=0,
        help="Overrides default per-model number of epoch setting. "
        "Defaults to 0 (per-model epoch & early-stopping).",
    )
    parser.addoption(
        "--eval-upto",
        action="store",
        default="train",
        choices=("train", "export", "optimize"),
        help="Choose train|export|optimize. Defaults to train.",
    )
    parser.addoption(
        "--data-root",
        action="store",
        default="data",
        help="Dataset root directory.",
    )
    parser.addoption(
        "--output-root",
        action="store",
        default="exp/perf",
        help="Output root directory.",
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution.",
    )


@pytest.fixture(scope="session")
def fxt_output_root(request: pytest.FixtureRequest) -> Path:
    """Output root + date + short commit hash."""
    output_root = request.config.getoption("--output-root")
    data_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit_str = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    return Path(output_root) / (data_str + "-" + commit_str)


@pytest.fixture
def fxt_model_id(request: pytest.FixtureRequest) -> str:
    """Skip by model category."""
    model_type: str = request.config.getoption("--model-type")
    model_template: ModelTemplate = request.param
    if model_type == "default":
        if model_template.model_category == ModelCategory.OTHER:
            pytest.skip(f"{model_template.model_category} category model")
    return model_template.model_template_id


@pytest.fixture
def fxt_benchmark(request: pytest.FixtureRequest, fxt_output_root: Path) -> OTXBenchmark:
    """Configure benchmark."""
    # Skip by dataset size
    data_size_option: str = request.config.getoption("--data-size")
    data_size: str = request.param[0]
    if data_size_option != "all":
        if data_size_option != data_size:
            pytest.skip(f"{data_size} datasets")

    # Options
    cfg: dict = request.param[1].copy()

    tags = cfg.get("tags", {})
    tags["data_size"] = data_size
    cfg["tags"] = tags

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
    cfg["output_root"] = str(fxt_output_root)
    cfg["dry_run"] = request.config.getoption("--dry-run")

    # Create benchmark
    benchmark = OTXBenchmark(
        **cfg,
    )

    return benchmark


@pytest.fixture(scope="session", autouse=True)
def fxt_benchmark_summary(fxt_output_root: Path):
    """Summarize all results at the end of test session."""
    yield
    all_results = OTXBenchmark.load_result(fxt_output_root)
    if all_results is not None:
        print("=" * 20, "[Benchmark summary]")
        print(all_results)
        output_path = fxt_output_root / "benchmark-summary.csv"
        all_results.to_csv(output_path, index=False)
        print(f"  -> Saved to {output_path}.")
