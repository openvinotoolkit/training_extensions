# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import json
import os
from collections.abc import Generator
from pathlib import Path

from behave import fixture, use_fixture
from behave.runner import Context

from tests.bdd.server_runner import DockerRunner, ProcessRunner

_FAILURES_REPORT_PATH = Path(__file__).parent / "timm_training_failures.json"


@fixture
def fastapi_server(context: Context) -> Generator[None]:
    """Fixture that uses the selected strategy."""
    runner_type = os.getenv("RUNNER", "process").lower()
    if runner_type not in ("docker", "process"):
        raise RuntimeError("Environment variable RUNNER must be either unset or set to 'docker' or 'process'")

    runner = DockerRunner(context) if runner_type == "docker" else ProcessRunner(context)
    runner.setup()
    try:
        runner.start_server()
        runner.wait_for_health()
        yield
    finally:
        try:
            runner.stop_server()
        finally:
            if os.getenv("KEEP_ARTIFACTS", "0").lower() not in ("1", "true"):
                runner.cleanup()


def before_all(context: Context) -> None:
    """Set up the server before each scenario."""
    context.failures = []
    use_fixture(fastapi_server, context)


def after_all(context: Context) -> None:
    """Dump collected model-architecture failures (stacktraces) to a JSON report."""
    failures = getattr(context, "failures", [])
    if failures:
        _FAILURES_REPORT_PATH.write_text(json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8")
