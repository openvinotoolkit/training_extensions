# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train, test, export, and optionally quantize a getitune model."""

from __future__ import annotations

import argparse
from pathlib import Path

from getitune.backend.lightning.engine import LightningEngine
from getitune.engine import create_engine
from getitune.types import TaskType
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision


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


def _disable_early_stopping(engine: LightningEngine) -> None:
    if not isinstance(engine, LightningEngine):
        return

    from lightning.pytorch.callbacks import EarlyStopping

    callbacks = engine._cache.args.get("callbacks", [])  # noqa: SLF001
    callbacks = [callback for callback in callbacks if not isinstance(callback, EarlyStopping)]
    engine._cache.args["callbacks"] = callbacks  # noqa: SLF001


def run(args: argparse.Namespace) -> None:
    """Run the requested training workflow."""
    task = args.task
    export_format_name = "openvino" if args.export_format == "ov" else args.export_format
    if args.quantize and export_format_name != "openvino":
        message = "quantization requires --export-format openvino"
        raise ValueError(message)

    engine = create_engine(
        model=args.model,
        data=str(args.data_root),
        work_dir=str(args.work_dir),
        device=args.device,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        task=task,
    )

    if args.no_early_stopping:
        _disable_early_stopping(engine)

    train_args = {"max_epochs": args.epochs, "precision": args.precision}
    if args.batch is not None and not isinstance(engine, LightningEngine):
        train_args["batch"] = args.batch
    if args.no_early_stopping and not isinstance(engine, LightningEngine):
        if hasattr(engine, "_training_defaults"):
            engine._training_defaults["patience"] = None  # noqa: SLF001
        else:
            train_args["patience"] = 0

    print(f"Training {args.model} for {args.epochs} epoch(s) on {args.device}")
    print(f"Train metrics: {engine.train(**train_args)}")
    print(f"Test metrics: {engine.test()}")

    if not (args.export or args.quantize):
        return

    export_format = ExportFormat.OPENVINO if export_format_name == "openvino" else ExportFormat.ONNX
    export_precision = Precision.FP16 if args.export_precision == "fp16" else Precision.FP32
    exported = engine.export(export_format=export_format, export_precision=export_precision)
    print(f"Exported {export_format.value}: {exported}")

    if args.quantize:
        quantized_engine = create_engine(
            model=exported,
            data=engine.datamodule,
            work_dir=str(args.work_dir),
        )
        quantized = quantized_engine.optimize()
        print(f"Quantized OpenVINO model: {quantized}")
        print(f"Quantized test metrics: {quantized_engine.test()}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name or recipe YAML path")
    parser.add_argument("--task", type=_task, help="Task type, for example DETECTION or INSTANCE_SEGMENTATION")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root or supported dataset file")
    parser.add_argument("--work-dir", type=Path, default=Path("./getitune-workspace"))
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint for warm-start training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, help="Optional batch-size override")
    parser.add_argument("--precision", default="16-mixed", help="Training precision, for example 32 or bf16")
    parser.add_argument("--device", default="auto", help="Device, for example auto, cpu, gpu, or xpu")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping configured by the recipe",
    )
    parser.add_argument("--export", action="store_true", help="Export the trained model")
    parser.add_argument("--export-format", choices=("openvino", "ov", "onnx"), default="openvino")
    parser.add_argument("--export-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--quantize", action="store_true", help="Quantize the exported OpenVINO model to INT8")
    return parser


def main() -> None:
    """Parse arguments and run training."""
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        message = f"error: {error}"
        raise SystemExit(message) from error


if __name__ == "__main__":
    main()
