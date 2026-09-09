# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export a getitune model to ONNX or OpenVINO with the requested precision."""

from __future__ import annotations

import argparse
from pathlib import Path

from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision


def _task(value: str) -> str:
    value = value.strip().upper().replace("-", "_").replace(" ", "_")
    from getitune.types import TaskType

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


def _quantize(model: Path, args: argparse.Namespace) -> Path:
    quantize_engine = create_engine(
        model=str(model),
        data=str(args.data_root),
        work_dir=str(args.work_dir),
    )
    return Path(quantize_engine.optimize())


def run(args: argparse.Namespace) -> Path:
    """Export a model to the requested format and precision."""
    if Path(str(args.model)).suffix.lower() in {".xml", ".onnx"}:
        message = (
            "--model must be a model name or recipe YAML when using export.py; "
            "exporting from an already-exported .xml/.onnx model is not supported."
        )
        raise ValueError(message)

    if args.precision == "int8" and args.format != "openvino":
        message = "INT8 quantization requires --format openvino"
        raise ValueError(message)

    if args.precision == "int8" and args.data_root is None:
        message = "--data-root is required to quantize a model to INT8"
        raise ValueError(message)

    engine = create_engine(
        model=args.model,
        data=str(args.data_root),
        work_dir=str(args.work_dir),
        device=args.device,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        task=args.task,
    )

    export_format = ExportFormat.ONNX if args.format == "onnx" else ExportFormat.OPENVINO
    export_precision = Precision.FP16 if args.precision == "fp16" else Precision.FP32
    exported = engine.export(export_format=export_format, export_precision=export_precision)
    print(f"Exported {export_format.value} {export_precision.value}: {exported}")

    if args.precision == "int8":
        quantized = _quantize(Path(exported), args)
        print(f"Quantized OpenVINO INT8: {quantized}")
        return Path(quantized)

    return Path(exported)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name or recipe YAML path")
    parser.add_argument("--task", type=_task, help="Task type, for example DETECTION or INSTANCE_SEGMENTATION")
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Dataset root, required for recipe export and INT8 quantization"
    )
    parser.add_argument("--checkpoint", type=Path, help="Trained model checkpoint to export")
    parser.add_argument("--work-dir", type=Path, default=Path("./getitune-workspace"))
    parser.add_argument("--format", choices=("openvino", "onnx"), default="openvino")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
    parser.add_argument("--device", default="auto", help="Device, for example auto, cpu, gpu, or xpu")
    return parser


def main() -> None:
    """Parse arguments and export the model."""
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        message = f"error: {error}"
        raise SystemExit(message) from error


if __name__ == "__main__":
    main()
