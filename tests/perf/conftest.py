# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import platform
import os
import re
import subprocess
import shutil
from cpuinfo import get_cpu_info
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from otx import __version__ as VERSION
from .benchmark import Benchmark

log = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add custom options for perf tests."""
    parser.addoption(
        "--benchmark-type",
        action="store",
        default="accuracy",
        choices=("accuracy", "efficiency", "all"),
        help="Choose accuracy|efficiency|all. Defaults to accuracy.",
    )
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
        help="Sign-off the user name who launched the regression tests this time, " 'e.g., `--user-name "John Doe"`.',
    )
    parser.addoption(
        "--mlflow-tracking-uri",
        type=str,
        help="URI for MLFlow Tracking server to store the regression test results.",
    )


@pytest.fixture(scope="session")
def fxt_benchmark_type(request: pytest.FixtureRequest) -> str:
    """Select benchmark type."""
    benchmark_type: str = request.config.getoption("--benchmark-type")
    msg = f"{benchmark_type = }"
    log.info(msg)
    return benchmark_type


@pytest.fixture(scope="session")
def fxt_model_type(request: pytest.FixtureRequest) -> str:
    """Model type to run the benchmark."""
    model_type = request.config.getoption("--model-type")
    msg = f"{model_type = }"
    log.info(msg)
    return model_type


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
def fxt_output_root(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Output root + date + short commit hash."""
    output_root = request.config.getoption("--output-root")
    if output_root is None:
        output_root = tmp_path_factory.mktemp("otx-benchmark")
    data_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit_str = os.environ.get("GH_CTX_SHA", "unknown")
    output_root = Path(output_root) / (data_str + "-" + commit_str[:7])
    msg = f"{output_root = }"
    log.info(msg)
    return output_root


@pytest.fixture(scope="session")
def fxt_summary_csv(request: pytest.FixtureRequest, fxt_output_root: Path) -> Path:
    """Path to benchmark result summary csv file."""
    summary_csv = request.config.getoption("--summary-csv")
    if summary_csv is None:
        summary_csv = fxt_output_root / "benchmark-summary.csv"
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
def fxt_user_name(request: pytest.FixtureRequest) -> str:
    """User name to sign off the regression test execution."""
    user_name = request.config.getoption("--user-name")
    msg = f"{user_name = }"
    log.info(msg)
    return user_name


@pytest.fixture(scope="session")
def fxt_mlflow_tracking_uri(request: pytest.FixtureRequest) -> str:
    """MLFLow tracking server URI."""
    mlflow_tracking_uri = urlparse(
        request.config.getoption("--mlflow-tracking-uri"),
    ).geturl()
    msg = f"{mlflow_tracking_uri = }"
    log.info(msg)
    return mlflow_tracking_uri


@pytest.fixture(scope="session")
def fxt_working_branch() -> str:
    """Git branch name for the current HEAD."""
    working_branch = os.environ.get("GH_CTX_REF_NAME", "unknown")
    msg = f"{working_branch = }"
    log.info(msg)
    return branch


@pytest.fixture
def fxt_model(request: pytest.FixtureRequest, fxt_model_type) -> Benchmark.Model:
    """Skip models according to user options."""
    model: Benchmark.Model = request.param
    if fxt_model_type == "default":
        if model.type == "other":
            pytest.skip(f"{model.type} type model")
    return model


@pytest.fixture
def fxt_dataset(request: pytest.FixtureRequest, fxt_data_size) -> Benchmark.Data:
    """Skip datasets according to user options."""
    dataset: Benchmark.Dataset = request.param
    if fxt_data_size != "all":
        if dataset.size != fxt_data_size:
            pytest.skip(f"{dataset.size} size dataset")
    return dataset


@pytest.fixture(scope="session")
def fxt_tags(fxt_user_name) -> dict[str, str]:
    """Tag fields to record the machine and user executing this perf test."""
    tags = {
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
    log.info(tags)
    return tags


@pytest.fixture
def fxt_benchmark(
    request: pytest.FixtureRequest,
    fxt_benchmark_type: str,
    fxt_data_root: Path,
    fxt_output_root: Path,
    fxt_num_epoch: int,
    fxt_num_repeat: int,
    fxt_eval_upto: str,
    fxt_tags: dict[str, str],
    fxt_dry_run: bool,
    fxt_accelerator: str,
) -> Benchmark:
    """Configure benchmark."""
    benchmark_type: str = request.param["type"]
    if fxt_benchmark_type != "all":
        if benchmark_type != fxt_benchmark_type:
            pytest.skip(f"{benchmark_type} benchmark")

    num_epoch_override = fxt_num_epoch
    if num_epoch_override == 0:  # 0: use default
        if benchmark_type == "efficiency":
            num_epoch_override = 2

    benchmark_metrics: str = request.param["metrics"]
    tags = fxt_tags.copy()
    tags["benchmark_type"] = benchmark_type

    return Benchmark(
        data_root=fxt_data_root,
        output_root=fxt_output_root,
        metrics=benchmark_metrics,
        num_epoch=num_epoch_override,
        num_repeat=fxt_num_repeat,
        eval_upto=fxt_eval_upto,
        tags=tags,
        dry_run=fxt_dry_run,
        accelerator=fxt_accelerator,
    )


#def logging_perf_results_to_mlflow(
#    version: str, branch: str, git_hash: str, results: pd.DataFrame, client: "MlflowClient"
#):
#    class DummyDatasetSource(mlflow.data.DatasetSource):
#        @staticmethod
#        def _get_source_type():
#            return "dummy"
#
#    class DummyDataset(mlflow.data.Dataset):
#        def _to_dict(self, base_dict):
#            return {
#                "name": base_dict["name"],
#                "digest": base_dict["digest"],
#                "source": base_dict["source"],
#                "source_type": base_dict["source_type"],
#            }
#
#    exp_name = f"[{branch}] OTX Performance Benchmark"
#    exp = client.get_experiment_by_name(exp_name)
#    if exp is None:
#        exp_id = client.create_experiment(exp_name, tags={"Project": "OpenVINO Training Extensions", "Branch": branch})
#    else:
#        exp_id = exp.experiment_id
#
#    mlflow.set_experiment(experiment_id=exp_id)
#
#    rows = results.to_dict(orient="records")
#    for row in rows:
#        task = row.pop("task")
#        model = row.pop("model")
#        data = row.pop("data")
#        data = os.path.dirname(data)
#        data_sz = row.pop("data_size")
#        benchmark = row.pop("benchmark")
#        runs = client.search_runs(
#            exp_id,
#            filter_string=f"tags.task LIKE '%{task}%' AND "
#            f"tags.model LIKE '%{model}%' AND "
#            f"tags.data LIKE '%{data}%' AND "
#            f"tags.benchmark LIKE '%{benchmark}%'",
#        )
#        run = None
#        is_new_run = True
#        run_name = f"[{benchmark}] {task} | {model}"
#        if len(runs) == 0:
#            run = client.create_run(exp_id, run_name=run_name)
#        else:
#            is_new_run = False
#            run = runs[0]
#
#        with mlflow.start_run(run_id=run.info.run_id):
#            if is_new_run:
#                mlflow.set_tag("task", task)
#                mlflow.set_tag("model", model)
#                mlflow.set_tag("data", data)
#                mlflow.set_tag("benchmark", benchmark)
#                dat_src = DummyDatasetSource()
#                dataset = DummyDataset(dat_src, data, data_sz)
#                mlflow.log_input(dataset)
#            mlflow.set_tag("version", version)
#            mlflow.set_tag("git-hash", git_hash)
#            for k, v in row.items():
#                if isinstance(v, int) or isinstance(v, float):
#                    k = k.replace("(", "_")
#                    k = k.replace(")", "")
#                    k = k.replace("%", "percentage")
#                    history = client.get_metric_history(run.info.run_id, k)
#                    step = 0
#                    if len(history) > 0:
#                        step = history[-1].step + 1
#                    # set 'synchronous' to True to show the metric graph correctly
#                    mlflow.log_metric(k, v, step=step, synchronous=True)
#
#
#@pytest.fixture(scope="session", autouse=True)
#def fxt_benchmark_summary(request: pytest.FixtureRequest, fxt_output_root: Path, fxt_working_branch, fxt_mlflow_client):
#    """Summarize all results at the end of test session."""
#    yield
#    all_results = OTXBenchmark.load_result(fxt_output_root)
#    if all_results is not None:
#        print("=" * 20, "[Benchmark summary]")
#        print(all_results)
#        output_path = request.config.getoption("--summary-csv")
#        if not output_path:
#            output_path = fxt_output_root / "benchmark-summary.csv"
#        all_results.to_csv(output_path)
#        print(f"  -> Saved to {output_path}.")
#
#        if fxt_mlflow_client is None:
#            print(
#                "Tracking server is not configured. for logging results, "
#                "set 'MLFLOW_TRACKING_SERVER_URI' environment variable to server URI ."
#            )
#            return
#
#        # logging to the mlflow for 'develop' or 'releases/x.x.x' branch
#        if fxt_working_branch == "develop" or bool(re.match("^releases/[0-9]+\.[0-9]+\.[0-9]+$", fxt_working_branch)):
#            version = VERSION
#            git_hash = str(fxt_output_root).split("-")[-1]
#            logging_perf_results_to_mlflow(version, fxt_working_branch, git_hash, all_results, fxt_mlflow_client)
#
#    if os.environ.get("BENCHMARK_RESULTS_CLEAR", False):
#        shutil.rmtree(fxt_output_root)
#
#
#@pytest.fixture(scope="session")
#def fxt_benchmark_reference() -> pd.DataFrame | None:
#    """Load reference benchmark results with index."""
#    ref = pd.read_csv(Path(__file__).parent.resolve() / "benchmark-reference.csv")
#    if ref is not None:
#        ref.set_index(["benchmark", "task", "data_size", "model"], inplace=True)
#    return ref
#
#
#@pytest.fixture(scope="session")
#def fxt_check_benchmark_result(fxt_benchmark_reference: pd.DataFrame | None) -> Callable:
#    """Return result checking function with reference data."""
#
#    def check_benchmark_result(result: pd.DataFrame, key: Tuple, checks: List[Dict]):
#        if fxt_benchmark_reference is None:
#            print("No benchmark references loaded. Skipping result checking.")
#            return
#
#        def get_entry(data: pd.DataFrame, key: Tuple) -> pd.Series:
#            if key in data.index:
#                return data.loc[key]
#            return None
#
#        target_entry = get_entry(fxt_benchmark_reference, key)
#        if target_entry is None:
#            print(f"No benchmark reference for {key} loaded. Skipping result checking.")
#            return
#
#        result_entry = get_entry(result, key)
#        assert result_entry is not None
#
#        def compare(name: str, op: str, margin: float):
#            if name not in result_entry or result_entry[name] is None or np.isnan(result_entry[name]):
#                return
#            if name not in target_entry or target_entry[name] is None or np.isnan(target_entry[name]):
#                return
#            if op == "==":
#                assert abs(result_entry[name] - target_entry[name]) < target_entry[name] * margin
#            elif op == "<":
#                assert result_entry[name] < target_entry[name] * (1.0 + margin)
#            elif op == ">":
#                assert result_entry[name] > target_entry[name] * (1.0 - margin)
#
#        for check in checks:
#            compare(**check)
#
#    return check_benchmark_result


class PerfTestBase:
    """Base perf test structure."""

    def _test_perf(self,
        model: Benchmark.Model,
        dataset: Benchmark.Dataset,
        benchmark: Benchmark,
    ):
        result = benchmark.run(
            model=model,
            dataset=dataset,
        )
