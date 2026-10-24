# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate a performance-only Markdown report from structured benchmark results."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {
    "schema_version",
    "task",
    "model",
    "dataset",
    "scenario",
    "seed",
    "training_device",
    "training_batch_size",
    "openvino_device",
    "git_sha",
    "software",
    "fp16",
    "int8",
}


def _cell(value: object, digits: int = 2) -> str:
    """Format a numeric report value or its missing/error sentinel."""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value) if value in {"None", "Error"} else "None"


def _load_result(path: Path) -> dict[str, Any]:
    """Load one canonical performance result, rejecting incomplete results.

    Raises:
        ValueError: If the result cannot be read, is not a JSON object, is
            missing required metadata, has non-physical device names, or has
            an invalid training batch size.
    """
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read performance result {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(result, dict):
        msg = f"Performance result must be a JSON object: {path}"
        raise ValueError(msg)  # noqa: TRY004 - single failure contract for incomplete results
    missing = _REQUIRED_FIELDS - result.keys()
    if missing:
        msg = f"Performance result {path} is missing fields: {sorted(missing)}"
        raise ValueError(msg)
    result["_path"] = str(path)
    for precision, expected in (("fp16", "FP16"), ("int8", "INT8")):
        section = result.get(precision, {})
        required = {"precision", "inference_batch_size", "throughput_fps", "latency_ms"}
        if not isinstance(section, dict) or not required <= section.keys():
            logger.warning("Performance result %s has an incomplete %s section.", path, precision)
            result[precision] = {"_error": "Error"}
            continue
        if section["precision"] != expected:
            logger.warning("Performance result %s has invalid %s precision.", path, precision)
            section["_error"] = "Error"
        numeric = (section["inference_batch_size"], section["throughput_fps"], section["latency_ms"])
        if not all(isinstance(value, (int, float)) and value > 0 for value in numeric):
            logger.warning("Performance result %s has invalid %s measurements.", path, precision)
            section["_error"] = "Error"
    logical_names = {"", "unknown", "auto", "cpu", "gpu", "xpu", "cuda", "mps"}
    for field in ("training_device", "openvino_device"):
        value = str(result.get(field, "")).strip()
        if value.lower() in logical_names:
            msg = f"Performance result {path} has missing physical {field}."
            raise ValueError(msg)
    if not isinstance(result.get("training_batch_size"), int) or result["training_batch_size"] < 1:
        msg = f"Performance result {path} has missing training batch size."
        raise ValueError(msg)
    software = result.get("software", {})
    if not isinstance(software, dict):
        software = {}
    result["software"] = software
    return result


def _training_iter_ms(seed_dir: Path) -> float | str:
    """Return post-warmup training iteration time in milliseconds."""
    import csv

    metrics_files = sorted((seed_dir / "train").glob("csv/version_*/metrics.csv"))
    if not metrics_files:
        logger.debug("Training metrics not found for performance result: %s", seed_dir)
        return "None"
    with metrics_files[-1].open(newline="", encoding="utf-8") as stream:
        values = [float(row["train/iter_time"]) for row in csv.DictReader(stream) if row.get("train/iter_time")]
    if not values:
        logger.debug("train/iter_time is missing from %s", metrics_files[-1])
        return "None"
    measured = values[1:] or values
    return 1000 * sum(measured) / len(measured)


def _training_memory(seed_dir: Path) -> tuple[float | str, float | str]:
    """Return peak allocated GPU memory and peak RAM from training metrics."""
    import csv

    metrics_files = sorted((seed_dir / "train").glob("csv/version_*/metrics.csv"))
    if not metrics_files:
        return "None", "None"
    with metrics_files[-1].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def maximum(column: str) -> float | str:
        values: list[float] = []
        for row in rows:
            value = row.get(column)
            if value:
                try:
                    values.append(float(value))
                except ValueError:
                    continue
        return max(values) if values else "None"

    return maximum("gpu_mem_allocated_gib"), maximum("ram_mem_gib")


def _load_aggregated_metrics(root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Index ``aggregated.csv`` peak-resource metrics by task/model/dataset."""
    import csv

    aggregated = root / "aggregated.csv"
    if not aggregated.exists():
        return {}
    indexed: dict[tuple[str, str, str], dict[str, float]] = {}
    with aggregated.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            numeric: dict[str, float] = {}
            for column in ("current:training:gpu_mem", "current:training:ram_mem", "current:time/train/iter"):
                value = row.get(column)
                if value:
                    try:
                        numeric[column] = float(value)
                    except ValueError:
                        continue
            indexed[(row.get("task", ""), row.get("model", ""), row.get("dataset", ""))] = numeric
    return indexed


def _first_number(*candidates: object) -> float | str:
    """Return the first strictly positive numeric candidate."""
    for candidate in candidates:
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and candidate > 0:
            return float(candidate)
    return "None"


def _input_shape(result: dict[str, Any]) -> str:
    """Return a hidden compatibility key for the benchmark input shape."""
    return str(result.get("input_shape", result.get("fp16", {}).get("input_shape", "None")))


def _metric_value(section: object, key: str) -> float | str:
    """Return a numeric metric or a report-cell sentinel."""
    if not isinstance(section, dict):
        return "Error"
    value = section.get(key)
    return value if isinstance(value, (int, float)) and value > 0 else section.get("_error", "None")


def _mean(values: list[float | str]) -> float | str:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if numeric:
        return statistics.mean(numeric)
    return next((value for value in values if value in {"Error", "None"}), "None")


def generate_performance_report(result_roots: list[Path], output: Path) -> None:
    """Combine canonical seed results into a strict performance-only report."""
    rows: list[dict[str, Any]] = []
    for root in result_roots:
        files = sorted(root.glob("**/performance_result.json"))
        if not files:
            msg = f"No performance_result.json files found under {root}"
            raise ValueError(msg)
        aggregated = _load_aggregated_metrics(root)
        for path in files:
            result = _load_result(path)
            key = (str(result.get("task")), str(result.get("model")), str(result.get("dataset")))
            fallback = aggregated.get(key, {})
            gpu_memory_gib, ram_memory_gib = _training_memory(path.parent)
            iteration_seconds = fallback.get("current:time/train/iter")
            result["training_iter_ms"] = _first_number(
                _training_iter_ms(path.parent),
                iteration_seconds * 1000 if iteration_seconds is not None else None,
            )
            result["gpu_memory_mb"] = _first_number(
                result.get("gpu_memory_mb"),
                fallback.get("current:training:gpu_mem"),
                gpu_memory_gib * 1024 if isinstance(gpu_memory_gib, (int, float)) else None,
            )
            result["ram_memory_mb"] = _first_number(
                result.get("ram_memory_mb"),
                fallback.get("current:training:ram_mem"),
                ram_memory_gib * 1024 if isinstance(ram_memory_gib, (int, float)) else None,
            )
            if result["gpu_memory_mb"] == "None":
                logger.warning("GPU memory could not be resolved for %s.", path)
            if result["ram_memory_mb"] == "None":
                logger.warning("Peak RAM could not be resolved for %s.", path)
            if result["training_iter_ms"] == "None":
                logger.warning("Training iteration time could not be resolved for %s.", path)
            rows.append(result)

    model_groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["task"],
            row["model"],
            row["scenario"],
            row["training_device"],
            row["openvino_device"],
            row["training_batch_size"],
            row.get("fp16", {}).get("inference_batch_size", "None"),
            _input_shape(row),
        )
        model_groups[key].append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_rows in model_groups.values():
        representative = dict(model_rows[0])
        fp16 = dict(representative["fp16"])
        int8 = dict(representative["int8"])
        representative["training_iter_ms"] = _mean([row["training_iter_ms"] for row in model_rows])
        representative["gpu_memory_mb"] = _mean([row.get("gpu_memory_mb", "None") for row in model_rows])
        representative["ram_memory_mb"] = _mean([row.get("ram_memory_mb", "None") for row in model_rows])
        for section_name, section in (("fp16", fp16), ("int8", int8)):
            source_sections = [row.get(section_name, {}) for row in model_rows]
            section["throughput_fps"] = _mean([_metric_value(item, "throughput_fps") for item in source_sections])
            section["latency_ms"] = _mean([_metric_value(item, "latency_ms") for item in source_sections])
        representative["fp16"] = fp16
        representative["int8"] = int8
        representative["datasets_used"] = sorted({str(row["dataset"]) for row in model_rows})
        representative["seed_count"] = len(model_rows)
        grouped[str(representative["task"])].append(representative)

    lines = [
        "# GetiTune Performance Report",
        "",
        "> Performance-only report. Accuracy metrics are intentionally excluded.",
        "",
    ]
    lines.append("Metrics are arithmetic means across compatible datasets, seeds, and result roots.")
    lines.append("Source datasets are retained in the canonical JSON results.")
    lines.append("")
    environments: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["training_device"]), str(row["openvino_device"]))
        software = row.get("software", {})
        if not isinstance(software, dict):
            software = {}
        existing = environments.get(key)
        if existing is not None and existing != software:
            msg = f"Conflicting software versions for training/OpenVINO devices {key}."
            raise ValueError(msg)
        environments[key] = software

    lines.extend(
        [
            "## Software Environment",
            "",
            "| Training Device | OpenVINO Device | Python | PyTorch | OpenVINO |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for (training_device, openvino_device), software in sorted(environments.items()):
        lines.append(
            f"| {training_device} | {openvino_device} | {software.get('python', 'None')} | "
            f"{software.get('torch', 'None')} | {software.get('openvino', 'None')} |"
        )
    lines.append("")

    header = (
        "| Model | Training Device | OpenVINO Device | Train Batch | OV Inference Batch | "
        "Train Iteration (ms) | GPU Memory (MB) | Peak RAM (MB) | FP16 FPS | FP Latency (ms) | "
        "INT8 FPS | INT8 Latency (ms) |"
    )
    separator = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    for task, task_rows in sorted(grouped.items()):
        lines.extend([f"## {task}", "", header, separator])
        for row in sorted(task_rows, key=lambda item: (str(item["model"]), str(item["training_device"]))):
            fp16 = row.get("fp16", {})
            int8 = row.get("int8", {})
            if not isinstance(fp16, dict):
                fp16 = {}
            if not isinstance(int8, dict):
                int8 = {}
            inference_batch = fp16.get("inference_batch_size", "None")
            if inference_batch != int8.get("inference_batch_size", inference_batch):
                logger.warning("FP16 and INT8 inference batches differ for %s.", row["model"])
            lines.append(
                f"| {row['model']} | {row['training_device']} | {row['openvino_device']} | "
                f"{row.get('training_batch_size', 'None')} | {inference_batch} | "
                f"{_cell(row.get('training_iter_ms'))} | {_cell(row.get('gpu_memory_mb'))} | "
                f"{_cell(row.get('ram_memory_mb'))} | {_cell(fp16.get('throughput_fps'))} | "
                f"{_cell(fp16.get('latency_ms'))} | {_cell(int8.get('throughput_fps'))} | "
                f"{_cell(int8.get('latency_ms'))} |"
            )
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the performance report generator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_roots", type=Path, nargs="+", help="Benchmark result directories.")
    parser.add_argument("--output", type=Path, default=Path("performance_report.md"))
    args = parser.parse_args()
    generate_performance_report(args.result_roots, args.output)


if __name__ == "__main__":
    main()
