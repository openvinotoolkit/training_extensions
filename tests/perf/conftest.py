# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import os
import platform
import subprocess
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Callable, TYPE_CHECKING

from cpuinfo import get_cpu_info
from mlflow.client import MlflowClient
import numpy as np
import pandas as pd
import pytest
import yaml

from .benchmark import Benchmark


def pytest_addoption(parser):
    """Add custom options for perf tests."""
    parser.addoption(
        "--model-category",
        action="store",
        default="all",
        choices=("default", "all"),
        help="Choose default|all. Defaults to all.",
    )
    parser.addoption(
        "--data-group",
        action="store",
        default="all",
        choices=("small", "medium", "large", "all"),
        help="Choose small|medium|large|all. Defaults to all.",
    )
    parser.addoption(
        "--num-repeat",
        action="store",
        default=0,
        help="Overrides default per-data-group number of repeat setting. "
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
        help="Output root directory. Defaults to temp directory.",
    )
    parser.addoption(
        "--summary-csv",
        action="store",
        help="Path to output summary cvs file. Defaults to {output-root}/benchmark-summary.csv",
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution.",
    )
    parser.addoption(
        "--user-name",
        type=str,
        default="anonymous",
        help='Sign-off the user name who launched the regression tests this time, e.g., `--user-name "John Doe"`.',
    )
    parser.addoption(
        "--mlflow-tracking-uri",  # Currently set by MLFLOW_TRACKING_SERVER_URI env variable. To be fixed.
        type=str,
        help="URI for MLFlow Tracking server to store the regression test results.",
    )
    parser.addoption(
        "--otx-ref",
        type=str,
        default="__CURRENT_BRANCH_COMMIT__",
        help="Target OTX ref (tag / branch name / commit hash) on main repo to test. Defaults to the current branch. "
        "`pip install otx[full]@https://github.com/openvinotoolkit/training_extensions.git@{otx_ref}` will be executed before run, "
        "and reverted after run. Works only for v1.x assuming CLI compatibility.",
    )


@pytest.fixture(scope="session")
def fxt_current_date() -> str:
    tz = timezone(offset=timedelta(hours=9), name="Seoul")
    return datetime.now(tz=tz).strftime("%Y%m%d-%H%M%S")


@pytest.fixture(scope="session")
def fxt_otx_ref(request: pytest.FixtureRequest) -> str | None:
    otx_ref = request.config.getoption("--otx-ref")
    if otx_ref == "__CURRENT_BRANCH_COMMIT__":
        otx_ref = None

    if otx_ref:
        # Install specific version
        subprocess.run(
            ["pip", "install", f"otx[full]@git+https://github.com/openvinotoolkit/training_extensions.git@{otx_ref}"],
            check=True,
        )

    yield otx_ref

    if otx_ref:
        # Restore the current version
        subprocess.run(
            ["pip", "install", "-e", ".[full]"],
            check=True,
        )


@pytest.fixture(scope="session")
def fxt_version_tags(fxt_current_date: str, fxt_otx_ref: str) -> dict[str, str]:
    """Version / branch / commit info."""
    otx = importlib.import_module("otx")
    otx = importlib.reload(otx)  # To get re-installed OTX version
    version_str = otx.__version__
    try:
        branch_str = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
        )  # noqa: S603, S607
    except Exception:
        branch_str = os.environ.get("GH_CTX_REF_NAME", "unknown")
    try:
        commit_str = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        )  # noqa: S603, S607
    except Exception:
        commit_str = os.environ.get("GH_CTX_SHA", "unknown")
    version_tags = {
        "otx_version": version_str,
        "otx_ref": fxt_otx_ref or commit_str,
        "test_branch": branch_str,
        "test_commit": commit_str,
        "date": fxt_current_date,
    }
    return version_tags


@pytest.fixture(scope="session")
def fxt_tags(request: pytest.FixtureRequest, fxt_version_tags: dict[str, str]) -> dict[str, str]:
    """Tag fields to record the machine and user executing this perf test."""
    tags = {
        **fxt_version_tags,
        "user_name": request.config.getoption("--user-name"),
        "machine_name": platform.node(),
        "cpu_info": get_cpu_info()["brand_raw"],
        "accelerator_info": subprocess.check_output(
            ["nvidia-smi", "-L"],  # noqa: S603, S607
        )
        .decode()
        .strip(),
    }
    print(f"{tags = }")
    return tags


@pytest.fixture(scope="session")
def fxt_output_root(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, fxt_current_date: str
) -> Path:
    """Output root + date + short commit hash."""
    output_root = request.config.getoption("--output-root")
    if output_root is None:
        output_root = tmp_path_factory.mktemp("otx-benchmark")
    return Path(output_root) / fxt_current_date


@pytest.fixture
def fxt_model_id(request: pytest.FixtureRequest) -> str:
    """Skip by model category."""
    from otx.api.entities.model_template import ModelCategory, ModelTemplate

    model_category: str = request.config.getoption("--model-category")
    model_template: ModelTemplate = request.param
    if model_category == "default":
        if model_template.model_category == ModelCategory.OTHER:
            pytest.skip(f"{model_template.model_category} category model")
    return model_template.model_template_id


@pytest.fixture
def fxt_benchmark(
    request: pytest.FixtureRequest,
    fxt_output_root: Path,
    fxt_tags: dict[str, str],
    fxt_benchmark_reference: pd.DataFrame | None,
) -> Benchmark:
    """Configure benchmark."""
    # Skip by data group
    data_group_option: str = request.config.getoption("--data-group")
    data_group: str = request.param[0]
    if data_group_option != "all":
        if data_group_option != data_group:
            pytest.skip(f"{data_group} datasets")

    # Options
    cfg: dict = request.param[1].copy()

    tags = cfg.get("tags", {})
    tags["data_group"] = data_group
    tags.update(fxt_tags)
    cfg["tags"] = tags

    num_epoch_override: int = int(request.config.getoption("--num-epoch"))
    if num_epoch_override > 0:  # 0: use default
        cfg["num_epoch"] = num_epoch_override

    num_repeat_override: int = int(request.config.getoption("--num-repeat"))
    if num_repeat_override > 0:  # 0: use default
        cfg["num_repeat"] = num_repeat_override

    cfg["eval_upto"] = request.config.getoption("--eval-upto")
    cfg["data_root"] = request.config.getoption("--data-root")
    cfg["output_root"] = str(fxt_output_root / tags["task"] / data_group)
    cfg["dry_run"] = request.config.getoption("--dry-run")
    cfg["reference_results"] = fxt_benchmark_reference

    # Create benchmark
    benchmark = Benchmark(
        **cfg,
    )

    return benchmark


def _log_benchmark_results_to_mlflow(results: pd.DataFrame, tags: dict[str, str], client: MlflowClient):
    for index, result in results.iterrows():
        task, model, data_group, data = index
        exp_name = f"[Benchmark] {task} | {model} | {data_group} | {data}"
        exp_tags = {
            "task": task,
            "model": model,
            "data_group": data_group,
            "data": data,
        }
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            exp_id = client.create_experiment(exp_name, tags=exp_tags)
        else:
            exp_id = exp.experiment_id
            if exp.lifecycle_stage != "active":
                client.restore_experiment(exp_id)
        run_name = f"[{tags['date']} | {tags['user_name']} | {tags['otx_version']} | {tags['test_branch']} | {tags['test_commit']}"
        run_tags = {k: v for k, v in result.items() if isinstance(v, str)}
        run_tags.update(**exp_tags, **tags)
        run = client.create_run(exp_id, run_name=run_name, tags=run_tags)
        run_metrics = {k: v for k, v in result.items() if not isinstance(v, str)}
        for k, v in run_metrics.items():
            k = k.replace("(", "_")
            k = k.replace(")", "")
            k = k.replace("%", "percentage")
            client.log_metric(run.info.run_id, k, v)


@pytest.fixture(scope="session", autouse=True)
def fxt_benchmark_summary(
    request: pytest.FixtureRequest, fxt_output_root: Path, fxt_tags: dict[str, str], fxt_mlflow_client: MlflowClient
):
    """Summarize all results at the end of test session."""
    yield
    raw_results = Benchmark.load_result(fxt_output_root)
    if raw_results is None:
        print("No benchmark results loaded in ", fxt_output_root)
        return

    print("=" * 20, "[Benchmark summary]")
    summary_results = [
        Benchmark.average_result(raw_results, ["task", "model", "data_group", "data"]),
        Benchmark.average_result(raw_results, ["task", "model", "data_group"]),
        Benchmark.average_result(raw_results, ["task", "model"]),
        Benchmark.average_result(raw_results, ["task"]),
    ]
    summary_results = pd.concat(summary_results)

    print("=" * 20, "[Benchmark summary]")
    print(summary_results)

    summary_csv = request.config.getoption("--summary-csv")
    if not summary_csv:
        summary_csv = fxt_output_root / "perf-benchmark-summary.csv"
    else:
        summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_results.to_csv(summary_csv)
    raw_results.to_csv(summary_csv.parent / "perf-benchmark-raw.csv")
    print(f"  -> Saved to {summary_csv}.")

    if fxt_mlflow_client is None:
        print(
            "Tracking server is not configured. for logging results, "
            "set 'MLFLOW_TRACKING_SERVER_URI' environment variable to server URI ."
        )
        return

    # NOTE: decide whether to enable mlflow logging only for 'develop' or 'releases/x.x.x' branch
    # test_branch = fxt_tags["test_branch"]
    # if test_branch == "develop" or bool(re.match("^releases/[0-9]+\.[0-9]+\.[0-9]+$", test_branch)):
    try:
        _log_benchmark_results_to_mlflow(summary_results, fxt_tags, fxt_mlflow_client)
    except Exception as e:
        print("MLFlow loging failed: ", e)

    if os.environ.get("BENCHMARK_RESULTS_CLEAR", False):
        shutil.rmtree(fxt_output_root)


@pytest.fixture(scope="session")
def fxt_benchmark_reference() -> pd.DataFrame | None:
    """Load reference benchmark results with index."""
    ref = pd.read_csv(Path(__file__).parent.resolve() / "benchmark-reference.csv")
    if ref is not None:
        ref = ref.set_index(["task", "model", "data_group", "data"])
    return ref
