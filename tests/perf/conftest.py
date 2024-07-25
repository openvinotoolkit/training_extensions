# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

import pytest
from cpuinfo import get_cpu_info
from mlflow.client import MlflowClient

from .benchmark import Benchmark
from .history import summary

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fxt_model_category(request: pytest.FixtureRequest) -> str:
    """Model category to run the benchmark."""
    model_category = request.config.getoption("--model-category")
    msg = f"{model_category = }"
    log.info(msg)
    return model_category


@pytest.fixture(scope="session")
def fxt_data_group(request: pytest.FixtureRequest) -> str:
    """Data group to run the benchmark."""
    data_group = request.config.getoption("--data-group")
    msg = f"{data_group = }"
    log.info(msg)
    return data_group


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
def fxt_otx_ref(request: pytest.FixtureRequest) -> str | None:
    otx_ref = request.config.getoption("--otx-ref")
    if otx_ref == "__CURRENT_BRANCH_COMMIT__":
        otx_ref = None

    if otx_ref:
        # Install specific version
        cmd = ["pip", "install", f"otx@git+https://github.com/openvinotoolkit/training_extensions.git@{otx_ref}"]
        subprocess.run(cmd, check=True)  # noqa: S603
        cmd = ["otx", "install"]
        subprocess.run(cmd, check=True)  # noqa: S603

    msg = f"{otx_ref = }"
    log.info(msg)
    yield otx_ref

    if otx_ref:
        # Restore the current version
        cmd = ["pip", "install", "-e", "."]
        subprocess.run(cmd, check=True)  # noqa: S603
        cmd = ["otx", "install"]
        subprocess.run(cmd, check=True)  # noqa: S603


@pytest.fixture(scope="session")
def fxt_version_tags(fxt_current_date: str, fxt_otx_ref: str) -> dict[str, str]:
    """Version / branch / commit info."""
    try:
        version_str = subprocess.check_output(["otx", "--version"]).decode("ascii").strip()[4:]  # noqa: S603, S607
    except Exception:
        version_str = "unknown"
    try:
        branch_str = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()  # noqa: S603, S607
    except Exception:
        branch_str = os.environ.get("GH_CTX_REF_NAME", "unknown")
    try:
        commit_str = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()  # noqa: S603, S607
    except Exception:
        commit_str = os.environ.get("GH_CTX_SHA", "unknown")
    version_tags = {
        "otx_version": version_str,
        "otx_ref": fxt_otx_ref or commit_str,
        "test_branch": branch_str,
        "test_commit": commit_str,
        "date": fxt_current_date,
    }
    msg = f"{version_tags = }"
    log.info(msg)
    return version_tags


@pytest.fixture(scope="session")
def fxt_summary_file(request: pytest.FixtureRequest, fxt_output_root: Path) -> Path:
    """Path to benchmark result summary csv file."""
    summary_file = request.config.getoption("--summary-file")
    summary_file = fxt_output_root / "benchmark-summary.csv" if summary_file is None else Path(summary_file)
    msg = f"{summary_file = }"
    log.info(msg)
    return summary_file


@pytest.fixture(scope="session")
def fxt_dry_run(request: pytest.FixtureRequest) -> str:
    """Option to print OTX commands without execution."""
    dry_run = request.config.getoption("--dry-run")
    msg = f"{dry_run = }"
    log.info(msg)
    return dry_run


@pytest.fixture(scope="session")
def fxt_deterministic(request: pytest.FixtureRequest) -> bool:
    """Option to turn on deterministic training."""
    deterministic_options = {
        "true": True,
        "false": False,
        "warn": "warn",
    }
    deterministic = request.config.getoption("--deterministic")
    if deterministic is None:
        return False
    deterministic = deterministic_options[deterministic]
    log.info(f"{deterministic=}")
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
    if fxt_model_category == "all":
        return model
    if fxt_model_category == "default":
        if model.category == "other":
            pytest.skip(f"{model.category} category model")
    elif fxt_model_category != model.category:
        pytest.skip(f"{model.category} category model")
    return model


@pytest.fixture()
def fxt_dataset(request: pytest.FixtureRequest, fxt_data_group) -> Benchmark.Data:
    """Skip datasets according to user options."""
    dataset: Benchmark.Dataset = request.param
    if fxt_data_group not in {"all", dataset.group}:
        pytest.skip(f"{dataset.group} group dataset")
    return dataset


@pytest.fixture(scope="session")
def fxt_tags(fxt_user_name: str, fxt_version_tags: dict[str, str], fxt_accelerator: str) -> dict[str, str]:
    """Tag fields to record the machine and user executing this perf test."""
    tags = {
        **fxt_version_tags,
        "user_name": fxt_user_name,
        "machine_name": platform.node(),
        "cpu_info": get_cpu_info()["brand_raw"],
    }
    if fxt_accelerator == "gpu":
        tags["accelerator_info"] = subprocess.check_output(["nvidia-smi", "-L"]).decode().strip()  # noqa: S603, S607
    elif fxt_accelerator == "xpu":
        raw = subprocess.check_output(args=["xpu-smi", "discovery", "--dump", "1,2"]).decode().strip()
        tags["accelerator_info"] = "\n".join([ret.replace('"', "").replace(",", " : ") for ret in raw.split("\n")[1:]])
    elif fxt_accelerator == "cpu":
        tags["accelerator_info"] = "cpu"
    msg = f"{tags = }"
    log.info(msg)
    return tags


@pytest.fixture(scope="session")
def fxt_resume_from(request: pytest.FixtureRequest) -> Path | None:
    resume_from = request.config.getoption("--resume-from")
    if resume_from is not None:
        resume_from = Path(resume_from)
    msg = f"{resume_from = }"
    log.info(msg)
    return resume_from


@pytest.fixture(scope="session")
def fxt_test_only(request: pytest.FixtureRequest) -> Literal["all", "train", "export", "optimize"] | None:
    test_only = request.config.getoption("--test-only")
    msg = f"{test_only = }"
    log.info(msg)
    return test_only


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
    fxt_resume_from: Path | None,
    fxt_test_only: Literal["all", "train", "export", "optimize"] | None,
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
        resume_from=fxt_resume_from,
        test_only=fxt_test_only,
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_benchmark_summary(
    fxt_output_root: Path,
    fxt_summary_file: Path,
    fxt_mlflow_client: MlflowClient,
    fxt_tags: dict[str, str],
):
    """Summarize all results at the end of test session."""
    yield

    raw_results = Benchmark.load_result(fxt_output_root)
    if raw_results is None or len(raw_results) == 0:
        print("No benchmark results loaded in ", fxt_output_root)
        return

    summary_results = summary.summarize(raw_results)

    print("=" * 20, "[Benchmark summary]")
    print(summary_results)
    fxt_summary_file.parent.mkdir(parents=True, exist_ok=True)
    raw_results.to_csv(fxt_summary_file.parent / "perf-benchmark-raw.csv", index=False)
    if fxt_summary_file.suffix == ".xlsx":
        summary_results.to_excel(fxt_summary_file)
    else:
        if fxt_summary_file.suffix != ".csv":
            print(f"{fxt_summary_file.suffix} output is not supported.")
            fxt_summary_file = fxt_summary_file.with_suffix(".csv")
        summary_results.to_csv(fxt_summary_file)
    print(f"  -> Saved to {fxt_summary_file}.")

    if fxt_mlflow_client:
        try:
            _log_benchmark_results_to_mlflow(raw_results, fxt_mlflow_client, fxt_tags)
        except Exception as e:
            print("MLFlow logging failed: ", e)


def _log_benchmark_results_to_mlflow(results: pd.DataFrame, client: MlflowClient, tags: dict[str, str]) -> None:
    results = summary.average(results, keys=["task", "model", "data_group", "data"])  # Average out seeds
    results = results.set_index(["task", "data_group", "data"])
    for index, result in results.iterrows():
        task, data_group, data = index
        model = result["model"]
        exp_name = f"[Benchmark] {task} | {data_group} | {data}"
        exp_tags = {
            "task": task,
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
        run_name = f"[{model}] {tags['date']} | {tags['user_name']} | {tags['otx_version']} | {tags['test_branch']} | {tags['test_commit']}"
        run_tags = {k: v for k, v in result.items() if isinstance(v, str)}
        run_tags.update(**exp_tags, **tags)
        run = client.create_run(exp_id, run_name=run_name, tags=run_tags)
        run_metrics = {k: v for k, v in result.items() if not isinstance(v, str)}
        for k, v in run_metrics.items():
            client.log_metric(run.info.run_id, k, v)


@pytest.fixture(scope="session")
def fxt_benchmark_reference() -> pd.DataFrame | None:
    """Load reference benchmark results with index."""
    ref = summary.load(Path(__file__).parent.resolve() / "history/v1.5.2", need_normalize=True)
    if ref is not None:
        ref = ref.set_index(["task", "model", "data_group", "data"])
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
