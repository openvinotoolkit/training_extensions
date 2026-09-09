# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a getitune model on a dataset and print the computed metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from torchmetrics import Metric, MetricCollection

from getitune.engine import create_engine
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.dice import _segm_callable
from getitune.metrics.fmeasure import _f_measure_callable
from getitune.metrics.mean_ap import _mask_rle_mean_ap_callable, _mean_ap_callable
from getitune.metrics.mlc_map import MultilabelmAP
from getitune.metrics.pck import _pck_measure_callable
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

MetricFactory = Callable[[LabelInfo], Metric | MetricCollection]


def _task(value: str) -> str:
    value = value.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "DETECTION": TaskType.DETECTION.value,
        "INSTANCE_SEGMENTATION": TaskType.INSTANCE_SEGMENTATION.value,
        "SEMANTIC_SEGMENTATION": TaskType.SEMANTIC_SEGMENTATION.value,
        "KEYPOINT_DETECTION": TaskType.KEYPOINT_DETECTION.value,
        "MULTI_CLASS_CLS": TaskType.MULTI_CLASS_CLS.value,
        "MULTI_LABEL_CLS": TaskType.MULTI_LABEL_CLS.value,
    }
    value = aliases.get(value, value)
    try:
        return TaskType(value).value
    except ValueError as error:
        choices = ", ".join(task.value for task in TaskType)
        message = f"unknown task {value!r}; choose from {choices}"
        raise argparse.ArgumentTypeError(message) from error


METRIC_REGISTRY: dict[str, dict[str, MetricFactory]] = {
    TaskType.MULTI_CLASS_CLS.value: {
        "accuracy": MultiClassClsMetricCallable,
        "f1-score": MultiClassClsMetricCallable,
    },
    TaskType.MULTI_LABEL_CLS.value: {
        "accuracy": MultiLabelClsMetricCallable,
        "f1-score": MultiLabelClsMetricCallable,
        "map": lambda label_info: MultilabelmAP(label_info),
    },
    TaskType.DETECTION.value: {
        "f1-score": _f_measure_callable,
        "map": _mean_ap_callable,
    },
    TaskType.INSTANCE_SEGMENTATION.value: {
        "f1-score": _f_measure_callable,
        "map": _mask_rle_mean_ap_callable,
    },
    TaskType.SEMANTIC_SEGMENTATION.value: {
        "dice": _segm_callable,
        "miou": _segm_callable,
    },
    TaskType.KEYPOINT_DETECTION.value: {
        "pck": _pck_measure_callable,
        "pck-score": _pck_measure_callable,
    },
}


def _resolve_metric(
    task: str, metric_names: list[str], label_info: LabelInfo
) -> Callable[[LabelInfo], Metric | MetricCollection]:
    available = METRIC_REGISTRY.get(task, {})
    unknown = [name for name in metric_names if name not in available]
    if unknown:
        supported = ", ".join(sorted(available)) or "none"
        message = f"unsupported metric(s) {unknown} for task {task}; supported: {supported}"
        raise ValueError(message)

    def _callable(current_label_info: LabelInfo) -> MetricCollection:
        metrics: dict[str, Metric] = {}
        for name in metric_names:
            result = available[name](current_label_info)
            if isinstance(result, MetricCollection):
                for key, metric in result.items():
                    metrics[f"{name}/{key}"] = metric
            else:
                metrics[name] = result
        return MetricCollection(metrics)

    _callable(label_info)

    return _callable


def run(args: argparse.Namespace) -> None:
    """Evaluate the model and print the test metrics."""
    model_path = Path(args.model)
    is_torch = model_path.suffix.lower() in {".ckpt", ".pt", ".pth"}
    if is_torch and args.checkpoint is None:
        args.checkpoint = model_path
        args.model = args.recipe

    if is_torch:
        if not args.recipe:
            message = "--recipe (model name or recipe path) is required for a torch checkpoint"
            raise ValueError(message)
        if not args.task:
            message = "--task is required for a torch checkpoint"
            raise ValueError(message)

    create_kwargs: dict = {
        "model": args.recipe if is_torch else str(model_path),
        "data": str(args.data_root),
        "work_dir": str(args.work_dir),
        "device": args.device,
    }
    if is_torch:
        create_kwargs["checkpoint"] = str(args.checkpoint)
        create_kwargs["task"] = args.task

    engine = create_engine(**create_kwargs)

    metric = None
    if args.metric:
        metric = _resolve_metric(engine.task, args.metric, engine.model.label_info)

    print(f"Evaluating {args.model} on {args.data_root} (task={engine.task})")
    metrics = engine.test(metric=metric)
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Trained model: .xml/.onnx file, or a .ckpt/.pt/.pth torch checkpoint",
    )
    parser.add_argument(
        "--recipe",
        help="Model name or recipe YAML path (required when --model is a torch checkpoint)",
    )
    parser.add_argument(
        "--task",
        type=_task,
        help="Task type, for example DETECTION (required for torch checkpoints)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Torch checkpoint path (defaults to --model when it is a checkpoint)",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Path to the dataset root used for evaluation",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("./getitune-workspace"))
    parser.add_argument("--device", default="auto", help="Device, for example auto, cpu, gpu, or xpu")
    parser.add_argument(
        "--metric",
        nargs="+",
        metavar="NAME",
        help=(
            "One or more metrics to evaluate (overrides recipe defaults). "
            "Supported per task: MULTI_CLASS_CLS=accuracy,f1-score; "
            "MULTI_LABEL_CLS=accuracy,f1-score,map; "
            "DETECTION=f1-score,map; INSTANCE_SEGMENTATION=f1-score,map; "
            "SEMANTIC_SEGMENTATION=dice,miou; KEYPOINT_DETECTION=pck,pck-score"
        ),
    )
    return parser


def main() -> None:
    """Parse arguments and evaluate the model."""
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        message = f"error: {error}"
        raise SystemExit(message) from error


if __name__ == "__main__":
    main()
