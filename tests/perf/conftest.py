# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import mlflow
import pandas as pd
import pytest
import yaml

from otx import __version__ as VERSION
from otx.api.entities.model_template import ModelCategory, ModelTemplate

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


@pytest.fixture(scope="session")
def fxt_output_root(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Output root + date + short commit hash."""
    output_root = request.config.getoption("--output-root")
    if output_root is None:
        output_root = tmp_path_factory.mktemp("otx-benchmark")
    data_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit_str = os.environ.get("GH_CTX_SHA", "unknown")
    print(f"Git SHA configured with {commit_str}")
    return Path(output_root) / (data_str + "-" + commit_str[:7])


@pytest.fixture(scope="session")
def fxt_working_branch() -> str:
    """Git branch name for the current HEAD."""
    branch = os.environ.get("GH_CTX_REF_NAME", "unknown")
    print(f"working branch name fixture configured with {branch}")
    return branch


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


def logging_perf_results_to_mlflow(
    version: str, branch: str, git_hash: str, results: pd.DataFrame, client: "MlflowClient"
):
    class DummyDatasetSource(mlflow.data.DatasetSource):
        @staticmethod
        def _get_source_type():
            return "dummy"

    class DummyDataset(mlflow.data.Dataset):
        def _to_dict(self, base_dict):
            return {
                "name": base_dict["name"],
                "digest": base_dict["digest"],
                "source": base_dict["source"],
                "source_type": base_dict["source_type"],
            }

    exp_name = f"[{branch}] OTX Performance Benchmark"
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name, tags={"Project": "OpenVINO Training Extensions", "Branch": branch})
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_id=exp_id)

    rows = results.to_dict(orient="records")
    for row in rows:
        task = row.pop("task")
        model = row.pop("model")
        data = row.pop("data")
        data = os.path.dirname(data)
        data_sz = row.pop("data_size")
        benchmark = row.pop("benchmark")
        runs = client.search_runs(
            exp_id,
            filter_string=f"tags.task LIKE '%{task}%' AND "
            f"tags.model LIKE '%{model}%' AND "
            f"tags.data LIKE '%{data}%' AND "
            f"tags.benchmark LIKE '%{benchmark}%'",
        )
        run = None
        is_new_run = True
        run_name = f"[{benchmark}] {task} | {model}"
        if len(runs) == 0:
            run = client.create_run(exp_id, run_name=run_name)
        else:
            is_new_run = False
            run = runs[0]

        with mlflow.start_run(run_id=run.info.run_id):
            if is_new_run:
                mlflow.set_tag("task", task)
                mlflow.set_tag("model", model)
                mlflow.set_tag("data", data)
                mlflow.set_tag("benchmark", benchmark)
                dat_src = DummyDatasetSource()
                dataset = DummyDataset(dat_src, data, data_sz)
                mlflow.log_input(dataset)
            mlflow.set_tag("version", version)
            mlflow.set_tag("git-hash", git_hash)
            for k, v in row.items():
                if isinstance(v, int) or isinstance(v, float):
                    k = k.replace("(", "_")
                    k = k.replace(")", "")
                    k = k.replace("%", "percentage")
                    history = client.get_metric_history(run.info.run_id, k)
                    step = 0
                    if len(history) > 0:
                        step = history[-1].step + 1
                    # set 'synchronous' to True to show the metric graph correctly
                    mlflow.log_metric(k, v, step=step, synchronous=True)


@pytest.fixture(scope="session", autouse=True)
def fxt_benchmark_summary(request: pytest.FixtureRequest, fxt_output_root: Path, fxt_working_branch, fxt_mlflow_client):
    """Summarize all results at the end of test session."""
    yield
    all_results = OTXBenchmark.load_result(fxt_output_root)
    if all_results is not None:
        print("=" * 20, "[Benchmark summary]")
        print(all_results)
        output_path = request.config.getoption("--summary-csv")
        if not output_path:
            output_path = fxt_output_root / "benchmark-summary.csv"
        all_results.to_csv(output_path, index=False)
        print(f"  -> Saved to {output_path}.")

        if fxt_mlflow_client is None:
            print(
                "Tracking server is not configured. for logging results, "
                "set 'MLFLOW_TRACKING_SERVER_URI' environment variable to server URI ."
            )
            return

        # logging to the mlflow for 'develop' or 'releases/x.x.x' branch
        if fxt_working_branch == "develop" or bool(re.match("^releases/[0-9]+\.[0-9]+\.[0-9]+$", fxt_working_branch)):
            version = VERSION
            git_hash = str(fxt_output_root).split("-")[-1]
            logging_perf_results_to_mlflow(version, fxt_working_branch, git_hash, all_results, fxt_mlflow_client)

    if os.environ.get("BENCHMARK_RESULTS_CLEAR", False):
        shutil.rmtree(fxt_output_root)
