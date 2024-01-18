# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import logging
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest
from cpuinfo import get_cpu_info
from otx import __version__

import mlflow

log = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--user-name",
        type=str,
        help="Sign-off the user name who launched the regression tests this time, " 'e.g., `--user-name "John Doe"`.',
    )
    parser.addoption(
        "--dataset-root-dir",
        type=Path,
        help="Dataset root directory path for the regression tests",
    )
    parser.addoption(
        "--mlflow-tracking-uri",
        type=str,
        help="URI for MLFlow Tracking server to store the regression test results.",
    )
    parser.addoption(
        "--num-repeat",
        type=int,
        default=1,
        help="The number of repetitions for each test case with different seed (default=1).",
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_user_name(request: pytest.FixtureRequest) -> str:
    """User name to sign off the regression test execution.

    This should be given by the PyTest CLI option.
    """
    user_name = request.config.getoption("--user-name", default=os.environ.get('USER_NAME'))
    if user_name is None:
        raise ValueError("Missing user name: Please provide it via the command-line option '--user-name' or set the 'USER_NAME' environment variable.")
    msg = f"user_name: {user_name}"
    log.info(msg)
    return user_name


@pytest.fixture(scope="session", autouse=True)
def fxt_dataset_root_dir(request: pytest.FixtureRequest) -> Path:
    """Dataset root directory path.

    This should be given by the PyTest CLI option.
    """
    dataset_root_dir = request.config.getoption("--dataset-root-dir", default=os.environ.get('CI_PERFORMANCE_DATASET_ROOT_DIR'))
    if dataset_root_dir is None:
        raise ValueError("Missing dataset root path: Please provide it via the command-line option '--dataset-root-dir' or set the 'CI_PERFORMANCE_DATASET_ROOT_DIR' environment variable.")
    dataset_root_path = Path(dataset_root_dir)
    msg = f"dataset_root_path: {dataset_root_path}"
    log.info(msg)
    return dataset_root_path


@pytest.fixture(scope="session", autouse=True)
def fxt_mlflow_tracking_uri(request: pytest.FixtureRequest) -> str:
    """MLFLow tracking server URI.

    This should be given by the PyTest CLI option.
    """
    mlflow_tracking_uri = request.config.getoption("--mlflow-tracking-uri", default=os.environ.get('MLFLOW_TRACKING_SERVER_URI'))
    if mlflow_tracking_uri is None:
        raise ValueError("Missing mlflow tracking server uri: Please provide it via the command-line option '--mlflow-tracking-uri' or set the 'MLFLOW_TRACKING_SERVER_URI' environment variable.")
    parsed_uri = urlparse(mlflow_tracking_uri).geturl()
    msg = f"fxt_mlflow_tracking_uri: {parsed_uri}"
    log.info(msg)
    return parsed_uri


@pytest.fixture(scope="session", autouse=True)
def fxt_num_repeat(request: pytest.FixtureRequest) -> int:
    """The number of repetition for each test case.

    The random seed will be set for [0, fxt_num_repeat - 1]. Default is one.
    """
    num_repeat = request.config.getoption("--num-repeat")
    msg = f"fxt_num_repeat: {fxt_num_repeat}"
    log.info(msg)
    return num_repeat


@pytest.fixture(scope="session", autouse=True)
def fxt_mlflow_experiment_name(fxt_user_name) -> str:
    """MLFlow Experiment name (unique key).

    MLFlow Experiment name is an unique key as same as experiment id.
    Every MLFlow Run belongs to MLFlow Experiment.
    """
    tz = timezone(offset=timedelta(hours=9), name="Seoul")
    date = datetime.now(tz=tz).date()
    return f"OTX: {__version__}, Signed-off-by: {fxt_user_name}, Date: {date}"


@pytest.fixture(scope="session", autouse=True)
def fxt_tags(fxt_user_name) -> dict[str, str]:
    """Tag fields to record the machine and user executing this regression test."""
    return {
        "user_name": fxt_user_name,
        "machine_name": platform.node(),
        "cpu_info": get_cpu_info()["brand_raw"],
        "accelerator_info": subprocess.check_output(
            ["nvidia-smi", "-L"],  # noqa: S603, S607
        )
        .decode()
        .strip(),
    }


@pytest.fixture(scope="session", autouse=True)
def fxt_mlflow_experiment(
    fxt_mlflow_experiment_name: str,
    fxt_mlflow_tracking_uri: str,
    fxt_tags: dict[str, str],
) -> None:
    """Set MLFlow Experiment

    If there is a MLFlow Experiment which has the same name with the given name,
    it will use that MLFlow Experiment. Otherwise, it will create a new one and use it.
    """
    mlflow.set_tracking_uri(fxt_mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(name=fxt_mlflow_experiment_name)
    exp_id = (
        mlflow.create_experiment(
            name=fxt_mlflow_experiment_name,
            tags=fxt_tags,
        )
        if exp is None
        else exp.experiment_id
    )
    mlflow.set_experiment(experiment_id=exp_id)


@pytest.fixture(scope="session", autouse=True)
def fxt_recipe_dir() -> Path:
    """OTX recipe directory."""
    import otx.recipe as otx_recipe

    return Path(otx_recipe.__file__).parent
