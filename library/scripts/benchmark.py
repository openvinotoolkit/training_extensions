# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark an existing model or export one from a getitune recipe first."""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
from pathlib import Path

from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision


def _input_shape(value: str) -> str:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        message = "input size must be written as HxW, for example 224x224"
        raise argparse.ArgumentTypeError(message)
    try:
        height, width = (int(part.strip()) for part in parts)
    except ValueError as error:
        message = "input size must contain two integers"
        raise argparse.ArgumentTypeError(message) from error
    if height <= 0 or width <= 0:
        message = "input size must be greater than zero"
        raise argparse.ArgumentTypeError(message)
    return f"[1,3,{height},{width}]"


def _benchmark_command(args: argparse.Namespace, model: Path, shape: str | None) -> list[str]:
    # benchmark_app receives a constructed argv list and does not use shell interpolation.
    command = [args.benchmark_app, "-m", str(model), "-d", args.device]
    if args.batch is not None:
        command.extend(["-b", str(args.batch)])
    if shape is not None:
        command.extend(["-shape", shape])
    if args.precision != "int8":
        command.extend(["-infer_precision", "f16" if args.precision == "fp16" else "f32"])
    else:
        command.extend(["-infer_precision", "i8"])
    if args.hint is not None:
        command.extend(["-hint", args.hint])
    if args.iterations is not None:
        command.extend(["-niter", str(args.iterations)])
    if args.time is not None:
        command.extend(["-t", str(args.time)])
    if args.infer_requests is not None:
        command.extend(["-nireq", str(args.infer_requests)])
    return command


def _quantize(model: Path, args: argparse.Namespace, data: str) -> Path:
    quantize_engine = create_engine(model=str(model), data=data, work_dir=str(args.work_dir))
    return quantize_engine.optimize()


def _prepare_model(args: argparse.Namespace) -> Path:
    model = Path(args.model)
    if model.suffix.lower() in {".xml", ".onnx"}:
        if not model.exists():
            message = f"model file not found: {model}"
            raise FileNotFoundError(message)
        if args.precision == "int8":
            if model.suffix.lower() == ".onnx":
                message = "INT8 quantization requires an OpenVINO XML model, not ONNX"
                raise ValueError(message)
            if args.data_root is None:
                message = "--data-root is required to quantize an existing model"
                raise ValueError(message)
            return _quantize(model, args, str(args.data_root))
        return model

    if args.data_root is None:
        message = "--data-root is required when --model is a model name or recipe path"
        raise ValueError(message)

    engine = create_engine(
        model=args.model,
        data=str(args.data_root),
        work_dir=str(args.work_dir),
        device=args.device,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        task=args.task,
    )
    if args.precision == "int8":
        exported = engine.export(export_format=ExportFormat.OPENVINO, export_precision=Precision.FP32)
        return _quantize(Path(exported), args, str(args.data_root))

    export_format = ExportFormat.ONNX if args.format == "onnx" else ExportFormat.OPENVINO
    export_precision = Precision.FP16 if args.precision == "fp16" else Precision.FP32
    return Path(engine.export(export_format=export_format, export_precision=export_precision))


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing .xml/.onnx model, model name, or recipe path")
    parser.add_argument("--data-root", type=Path, help="Dataset root, required for recipe export and quantization")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint used when exporting a recipe model")
    parser.add_argument("--task", help="Optional task used to disambiguate a model name")
    parser.add_argument("--work-dir", type=Path, default=Path("./getitune-workspace"))
    parser.add_argument("--format", choices=("openvino", "onnx"), default="openvino")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--device", default="CPU", help="OpenVINO device, for example CPU, GPU, or NPU")
    parser.add_argument("--batch", type=int)
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument(
        "--input-size",
        type=_input_shape,
        default=None,
        help="Optional input image size, for example 224x224 (default: model shape)",
    )
    shape.add_argument(
        "--shape",
        default=None,
        help="Optional benchmark_app input shape, for example [1,3,224,224] (default: model shape)",
    )
    parser.add_argument("--hint", choices=("latency", "throughput", "none"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--time", type=int)
    parser.add_argument("--infer-requests", type=int)
    parser.add_argument(
        "--benchmark-app",
        default="benchmark_app",
        help="Path to the OpenVINO benchmark_app executable",
    )
    return parser


def main() -> None:
    """Prepare a model and run OpenVINO benchmark_app."""
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = _prepare_model(args)
        shape = args.shape or args.input_size
        command = _benchmark_command(args, model, shape)
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)  # noqa: S603
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        message = f"error: {error}"
        raise SystemExit(message) from error


if __name__ == "__main__":
    main()
