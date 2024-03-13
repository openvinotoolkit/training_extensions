# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import pandas as pd
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pytest
from cpuinfo import get_cpu_info
from mlflow.client import MlflowClient

from .benchmark import Benchmark


log = logging.getLogger(__name__)


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
        "--deterministic",
        action="store_true",
        default=False,
        help="Turn on deterministic training.",
    )
    parser.addoption(
        "--user-name",
        type=str,
        default="anonymous",
        help='Sign-off the user name who launched the regression tests this time, e.g., `--user-name "John Doe"`.',
    )
    parser.addoption(
        "--mlflow-tracking-uri",
        type=str,
        help="URI for MLFlow Tracking server to store the regression test results.",
    )


@pytest.fixture(scope="session")
def fxt_model_category(request: pytest.FixtureRequest) -> str:
    """Model category to run the benchmark."""
    model_category = request.config.getoption("--model-category")
    msg = f"{model_category = }"
    log.info(msg)
    return model_category


@pytest.fixture(scope="session")
def fxt_data_size(request: pytest.FixtureRequest) -> str:
    """Data size to run the benchmark."""
    data_size = request.config.getoption("--data-size")
    msg = f"{data_size = }"
    log.info(msg)
    return data_size


@pytest.fixture(scope="session")
def fxt_num_repeat(request: pytest.FixtureRequest) -> int:
    """Number of repeated run with different random seed."""
    num_repeat = int(request.config.getoption("--num-repeat"))
    msg = f"{num_repeat = }"
    log.info(msg)
    return num_repeat


@pytest.fixture(scope="session")
def fxt_num_epoch(request: pytest.FixtureRequest) -> int:
    """Number of epochs to train models."""
    num_epoch = int(request.config.getoption("--num-epoch"))
    msg = f"{num_epoch = }"
    log.info(msg)
    return num_epoch


@pytest.fixture(scope="session")
def fxt_eval_upto(request: pytest.FixtureRequest) -> str:
    """Last operation to evaluate ~ train|export|optimize."""
    eval_upto = request.config.getoption("--eval-upto")
    msg = f"{eval_upto = }"
    log.info(msg)
    return eval_upto


@pytest.fixture(scope="session")
def fxt_data_root(request: pytest.FixtureRequest) -> Path:
    """Dataset root directory path."""
    data_root = Path(request.config.getoption("--data-root"))
    msg = f"{data_root = }"
    log.info(msg)
    return data_root


@pytest.fixture(scope="session")
def fxt_current_date() -> str:
    tz = timezone(offset=timedelta(hours=9), name="Seoul")
    return datetime.now(tz=tz).strftime("%Y%m%d-%H%M%S")


@pytest.fixture(scope="session")
def fxt_output_root(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    fxt_current_date: str,
) -> Path:
    """Output root + date + short commit hash."""
    output_root = request.config.getoption("--output-root")
    if output_root is None:
        output_root = tmp_path_factory.mktemp("otx-benchmark")
    output_root = Path(output_root) / fxt_current_date
    msg = f"{output_root = }"
    log.info(msg)
    return output_root


@pytest.fixture(scope="session")
def fxt_version_tags(fxt_current_date: str) -> dict[str, str]:
    """Version / branch / commit info."""
    import otx

    version_str = otx.__version__
    try:
        branch_str = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()  # noqa: S603, S607
    except Exception:
        branch_str = os.environ.get("GH_CTX_REF_NAME", "unknown")
    try:
        commit_str = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()  # noqa: S603, S607
    except Exception:
        commit_str = os.environ.get("GH_CTX_SHA", "unknown")
    version_tags = {
        "version": version_str,
        "branch": branch_str,
        "commit": commit_str,
        "date": fxt_current_date,
    }
    msg = f"{version_tags = }"
    log.info(msg)
    return version_tags


@pytest.fixture(scope="session")
def fxt_summary_csv(request: pytest.FixtureRequest, fxt_output_root: Path) -> Path:
    """Path to benchmark result summary csv file."""
    summary_csv = request.config.getoption("--summary-csv")
    summary_csv = fxt_output_root / "benchmark-summary.csv" if summary_csv is None else Path(summary_csv)
    msg = f"{summary_csv = }"
    log.info(msg)
    return summary_csv


@pytest.fixture(scope="session")
def fxt_dry_run(request: pytest.FixtureRequest) -> str:
    """Option to print OTX commands without execution."""
    dry_run = request.config.getoption("--dry-run")
    msg = f"{dry_run = }"
    log.info(msg)
    return dry_run


@pytest.fixture(scope="session")
def fxt_deterministic(request: pytest.FixtureRequest) -> str:
    """Option to turn on deterministic training."""
    deterministic = request.config.getoption("--deterministic")
    msg = f"{deterministic = }"
    log.info(msg)
    return deterministic


@pytest.fixture(scope="session")
def fxt_user_name(request: pytest.FixtureRequest) -> str:
    """User name to sign off the regression test execution."""
    user_name = request.config.getoption("--user-name")
    msg = f"{user_name = }"
    log.info(msg)
    return user_name


@pytest.fixture(scope="session")
def fxt_mlflow_client(request: pytest.FixtureRequest) -> MlflowClient:
    """MLFLow tracking client."""
    mlflow_tracking_uri = urlparse(
        request.config.getoption("--mlflow-tracking-uri"),
    ).geturl()
    msg = f"{mlflow_tracking_uri = }"
    log.info(msg)
    if mlflow_tracking_uri:
        return MlflowClient(mlflow_tracking_uri)
    return None


@pytest.fixture()
def fxt_model(request: pytest.FixtureRequest, fxt_model_category) -> Benchmark.Model:
    """Skip models according to user options."""
    model: Benchmark.Model = request.param
    if fxt_model_category == "default" and model.category == "other":
        pytest.skip(f"{model.category} category model")
    return model


@pytest.fixture()
def fxt_dataset(request: pytest.FixtureRequest, fxt_data_size) -> Benchmark.Data:
    """Skip datasets according to user options."""
    dataset: Benchmark.Dataset = request.param
    if fxt_data_size not in {"all", dataset.size}:
        pytest.skip(f"{dataset.size} size dataset")
    return dataset


@pytest.fixture(scope="session")
def fxt_tags(fxt_user_name: str, fxt_version_tags: dict[str, str]) -> dict[str, str]:
    """Tag fields to record the machine and user executing this perf test."""
    tags = {
        **fxt_version_tags,
        "user_name": fxt_user_name,
        "machine_name": platform.node(),
        "cpu_info": get_cpu_info()["brand_raw"],
        "accelerator_info": subprocess.check_output(
            ["nvidia-smi", "-L"],  # noqa: S603, S607
        )
        .decode()
        .strip(),
    }
    msg = f"{tags = }"
    log.info(msg)
    return tags


@pytest.fixture()
def fxt_benchmark(
    fxt_data_root: Path,
    fxt_output_root: Path,
    fxt_num_epoch: int,
    fxt_num_repeat: int,
    fxt_eval_upto: str,
    fxt_tags: dict[str, str],
    fxt_dry_run: bool,
    fxt_deterministic: bool,
    fxt_accelerator: str,
    fxt_benchmark_reference: pd.DataFrame | None,
) -> Benchmark:
    """Configure benchmark."""
    return Benchmark(
        data_root=fxt_data_root,
        output_root=fxt_output_root,
        num_epoch=fxt_num_epoch,
        num_repeat=fxt_num_repeat,
        eval_upto=fxt_eval_upto,
        tags=fxt_tags,
        dry_run=fxt_dry_run,
        deterministic=fxt_deterministic,
        accelerator=fxt_accelerator,
        reference_results=fxt_benchmark_reference,
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_benchmark_summary(
    fxt_output_root: Path,
    fxt_summary_csv: Path,
    fxt_mlflow_client: MlflowClient,
    fxt_tags: dict[str, str],
):
    """Summarize all results at the end of test session."""
    yield
    all_results = Benchmark.load_result(fxt_output_root)
    if all_results is not None:
        print("=" * 20, "[Benchmark summary]")
        print(all_results)
        fxt_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        all_results.to_csv(fxt_summary_csv)
        print(f"  -> Saved to {fxt_summary_csv}.")

        if fxt_mlflow_client:
            _log_benchmark_results_to_mlflow(all_results, fxt_mlflow_client, fxt_tags)


def _log_benchmark_results_to_mlflow(results: pd.DataFrame, client: MlflowClient, tags: dict[str, str]) -> None:
    for index, data in results.iterrows():
        task, data_size, model = index
        exp_name = f"[Benchmark] {task} | {model} | {data_size}"
        exp_tags = {
            "task": task,
            "model": model,
            "data_size": data_size,
        }
        exp = client.get_experiment_by_name(exp_name)
        exp_id = client.create_experiment(exp_name, tags=exp_tags) if not exp else exp.experiment_id
        if exp.lifecycle_stage != "active":
            client.restore_experiment(exp_id)
        run_name = f"[{tags['date']} | {tags['user_name']} | {tags['version']} | {tags['branch']} | {tags['commit']}"
        run_tags = {k: v for k, v in data.items() if isinstance(v, str)}
        run_tags.update(**exp_tags, **tags)
        run = client.create_run(exp_id, run_name=run_name, tags=run_tags)
        run_metrics = {k: v for k, v in data.items() if not isinstance(v, str)}
        for k, v in run_metrics.items():
            client.log_metric(run.info.run_id, k, v)


@pytest.fixture(scope="session")
def fxt_benchmark_reference() -> pd.DataFrame | None:
    """Load reference benchmark results with index."""
    ref = pd.read_csv(Path(__file__).parent.resolve() / "benchmark-reference.csv")
    if ref is not None:
        ref.set_index(["task", "data_size", "model"], inplace=True)
    return ref


class PerfTestBase:
    """Base perf test structure."""

    def _test_perf(
        self,
        model: Benchmark.Model,
        dataset: Benchmark.Dataset,
        benchmark: Benchmark,
        criteria: list[Benchmark.Criterion],
    ) -> None:
        result = benchmark.run(
            model=model,
            dataset=dataset,
            criteria=criteria,
        )
        benchmark.check(
            result=result,
            criteria=criteria,
        )
