# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


@pytest.fixture(scope="module", autouse=True)
def fxt_user_name(request: pytest.FixtureRequest) -> str:
    """User name to sign off the regression test execution.

    This should be given by the PyTest CLI option.
    """
    user_name = request.config.getoption("--user-name")
    msg = f"user_name: {user_name}"
    log.info(msg)
    return user_name


@pytest.fixture(scope="module", autouse=True)
def fxt_dataset_root_dir(request: pytest.FixtureRequest) -> Path:
    """Dataset root directory path.

    This should be given by the PyTest CLI option.
    """
    dataset_root_dir = request.config.getoption("--data-root")
    msg = f"dataset_root_dir: {dataset_root_dir}"
    log.info(msg)
    return dataset_root_dir


@pytest.fixture(scope="module", autouse=True)
def fxt_mlflow_tracking_uri(request: pytest.FixtureRequest) -> str:
    """MLFLow tracking server URI.

    This should be given by the PyTest CLI option.
    """
    mlflow_tracking_uri = urlparse(
        request.config.getoption("--mlflow-tracking-uri"),
    ).geturl()
    msg = f"fxt_mlflow_tracking_uri: {mlflow_tracking_uri}"
    log.info(msg)
    return mlflow_tracking_uri


@pytest.fixture(scope="module", autouse=True)
def fxt_num_repeat(request: pytest.FixtureRequest) -> int:
    """The number of repetition for each test case.

    The random seed will be set for [0, fxt_num_repeat - 1]. Default is one.
    """
    num_repeat = request.config.getoption("--num-repeat")
    msg = f"fxt_num_repeat: {fxt_num_repeat}"
    log.info(msg)
    return num_repeat


@pytest.fixture(scope="module", autouse=True)
def fxt_mlflow_experiment_name(fxt_user_name) -> str:
    """MLFlow Experiment name (unique key).

    MLFlow Experiment name is an unique key as same as experiment id.
    Every MLFlow Run belongs to MLFlow Experiment.
    """
    tz = timezone(offset=timedelta(hours=9), name="Seoul")
    date = datetime.now(tz=tz).date()
    return f"OTX: {__version__}, Signed-off-by: {fxt_user_name}, Date: {date}"


@pytest.fixture(scope="module", autouse=True)
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


@pytest.fixture(scope="module", autouse=True)
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


@pytest.fixture(scope="module", autouse=True)
def fxt_recipe_dir() -> Path:
    """OTX recipe directory."""
    import otx.recipe as otx_recipe

    return Path(otx_recipe.__file__).parent
