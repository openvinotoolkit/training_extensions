# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the standalone library scripts."""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from scripts import benchmark, export, train
from scripts import test as test_script

from getitune.types.label import LabelInfo


def test_predict_help_works_when_run_as_a_script() -> None:
    """The documented direct invocation must resolve the local COCO utilities."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "predict.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout


def test_resolve_metric_returns_fresh_collections() -> None:
    """Metric state must not be shared between separate engine configurations."""
    label_info = LabelInfo(
        label_names=["cat", "dog"],
        label_ids=["0", "1"],
        label_groups=[["cat", "dog"]],
    )
    metric_factory = test_script._resolve_metric("MULTI_CLASS_CLS", ["accuracy"], label_info)

    first = metric_factory(label_info)
    second = metric_factory(label_info)

    assert first is not second
    assert first["accuracy/accuracy"] is not second["accuracy/accuracy"]


@pytest.mark.parametrize("suffix", [".xml", ".onnx"])
def test_export_rejects_exported_models_before_engine_creation(mocker, suffix: str, tmp_path: Path) -> None:
    """Exporting an already-exported model must fail before create_engine is called."""
    create_engine = mocker.patch("scripts.export.create_engine")
    args = Namespace(
        model=tmp_path / f"model{suffix}",
        precision="fp32",
        format="openvino",
        data_root=tmp_path / "dataset",
        work_dir=tmp_path,
        device="auto",
        checkpoint=None,
        task=None,
    )

    with pytest.raises(ValueError, match="already-exported"):
        export.run(args)

    create_engine.assert_not_called()


def test_script_task_help_uses_supported_values() -> None:
    """CLI help must not advertise the unsupported OBJECT_DETECTION value."""
    help_text = export.build_parser().format_help() + train.build_parser().format_help()

    assert "INSTANCE_SEGMENTATION" in help_text
    assert "OBJECT DETECTION" not in help_text


def test_benchmark_command_is_argument_list() -> None:
    """benchmark_app receives separate arguments instead of shell-parsed text."""
    args = Namespace(
        benchmark_app="benchmark_app",
        device="CPU",
        batch=2,
        precision="fp16",
        hint="latency",
        iterations=10,
        time=None,
        infer_requests=1,
    )

    command = benchmark._benchmark_command(args, Path("model.xml"), "[1,3,224,224]")

    assert command == [
        "benchmark_app",
        "-m",
        "model.xml",
        "-d",
        "CPU",
        "-b",
        "2",
        "-shape",
        "[1,3,224,224]",
        "-infer_precision",
        "f16",
        "-hint",
        "latency",
        "-niter",
        "10",
        "-nireq",
        "1",
    ]
