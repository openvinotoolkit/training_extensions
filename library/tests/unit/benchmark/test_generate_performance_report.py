# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the strict performance-only report generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from getitune.benchmark.generate_performance_report import generate_performance_report


def _write_result(
    root: Path,
    *,
    seed: int = 2,
    scenario: str = "custom",
    throughput_fps: float = 100.0,
    latency_ms: float = 2.0,
    training_device: str = "NVIDIA GeForce RTX 3090",
    openvino_device: str = "Intel(R) Core(TM) i9-14900K",
) -> None:
    seed_dir = root / "classification" / "multi_class_cls" / "model_a" / "dataset_a" / scenario / str(seed)
    metrics = seed_dir / "train" / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("train/iter_time\n0.9\n0.1\n0.2\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "task": "classification/multi_class_cls",
        "model": "model_a",
        "dataset": "dataset_a",
        "scenario": scenario,
        "seed": seed,
        "training_device": training_device,
        "training_batch_size": 16,
        "gpu_memory_mb": 2048.0,
        "ram_memory_mb": 8192.0,
        "openvino_device": openvino_device,
        "openvino_target": "CPU",
        "git_sha": "abc123",
        "software": {"python": "3.12", "getitune": "1", "torch": "2", "openvino": "3", "nncf": "4"},
        "fp16": {
            "precision": "FP16",
            "inference_batch_size": 1,
            "throughput_fps": throughput_fps,
            "latency_ms": latency_ms,
        },
        "int8": {"precision": "INT8", "inference_batch_size": 1, "throughput_fps": 200.0, "latency_ms": 1.0},
    }
    (seed_dir / "performance_result.json").write_text(json.dumps(result), encoding="utf-8")


def test_generates_exact_performance_schema(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_result(root)
    output = tmp_path / "performance.md"

    generate_performance_report([root], output)

    report = output.read_text(encoding="utf-8")
    assert "| Model | Training Device | OpenVINO Device | Train Batch | OV Inference Batch |" in report
    assert "GPU Memory (MB) | Peak RAM (MB) | FP16 FPS | FP Latency (ms) | INT8 FPS | INT8 Latency (ms) |" in report
    assert "NVIDIA GeForce RTX 3090" in report
    assert "Intel(R) Core(TM) i9-14900K" in report
    assert "## Software Environment" in report
    assert "| Python | PyTorch | OpenVINO |" in report
    assert "| 3.12 | 2 | 3 |" in report
    assert (
        "| NVIDIA GeForce RTX 3090 | Intel(R) Core(TM) i9-14900K | 16 | 1 | 150.00 | 2048.00 | 8192.00 | 100.00 | 2.00 | 200.00 | 1.00 |"
        in report
    )
    assert "| Dataset |" not in report


def test_averages_multiple_seeds_into_one_row(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_result(root, seed=0, throughput_fps=100, latency_ms=2)
    _write_result(root, seed=1, throughput_fps=200, latency_ms=4)
    output = tmp_path / "performance.md"

    generate_performance_report([root], output)

    report = output.read_text(encoding="utf-8")
    assert report.count("| model_a |") == 1
    assert "| 150.00 | 3.00 |" in report


def test_same_model_on_different_devices_keeps_distinguishable_rows(tmp_path: Path) -> None:
    cpu_root = tmp_path / "cpu-results"
    _write_result(
        cpu_root,
        training_device="Intel(R) Core(TM) i9-14900K",
        openvino_device="Intel(R) Core(TM) i9-14900K",
    )
    xpu_root = tmp_path / "xpu-results"
    _write_result(
        xpu_root,
        training_device="Intel(R) Arc(TM) A770 Graphics",
        openvino_device="Intel(R) Arc(TM) A770 Graphics",
    )
    output = tmp_path / "performance.md"

    generate_performance_report([cpu_root, xpu_root], output)

    report = output.read_text(encoding="utf-8")
    assert report.count("| model_a |") == 2
    assert "| model_a | Intel(R) Core(TM) i9-14900K | Intel(R) Core(TM) i9-14900K |" in report
    assert "| model_a | Intel(R) Arc(TM) A770 Graphics | Intel(R) Arc(TM) A770 Graphics |" in report


def test_missing_required_metadata_fails(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_result(root)
    result_path = next(root.glob("**/performance_result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("openvino_device")
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        generate_performance_report([root], tmp_path / "report.md")


def test_logical_device_name_fails(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_result(root)
    result_path = next(root.glob("**/performance_result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["training_device"] = "xpu"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="missing physical training_device"):
        generate_performance_report([root], tmp_path / "report.md")


def test_conflicting_software_versions_fail(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_result(root, seed=0)
    _write_result(root, seed=1)
    result_path = sorted(root.glob("**/performance_result.json"))[1]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["software"]["torch"] = "different"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="Conflicting software versions"):
        generate_performance_report([root], tmp_path / "report.md")
