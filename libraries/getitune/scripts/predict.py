# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Predict with an OV / ONNX / torch model and save predictions in COCO format."""

from __future__ import annotations

import argparse
from pathlib import Path

from coco_utils import iter_image_files, write_coco

from getitune.engine import create_engine


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


def run(args: argparse.Namespace) -> None:
    """Run prediction and write COCO predictions to disk."""
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
        "data": str(args.input),
        "work_dir": str(args.work_dir),
        "device": args.device,
    }
    if is_torch:
        create_kwargs["checkpoint"] = str(args.checkpoint)
        create_kwargs["task"] = args.task

    engine = create_engine(**create_kwargs)

    image_files = iter_image_files(Path(args.input))
    predictions = engine.predict(data=str(args.input))

    write_coco(predictions, Path(args.output), engine.model.label_info, image_files)
    print(f"Wrote COCO predictions for {len(predictions)} batch(es) to {args.output}")


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
        "--input",
        required=True,
        type=Path,
        help="Dataset root or images folder to run predictions on",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("./getitune-workspace"))
    parser.add_argument("--device", default="auto", help="Device, for example auto, cpu, gpu, or xpu")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"), help="Output COCO JSON path")
    return parser


def main() -> None:
    """Parse arguments and run prediction."""
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        message = f"error: {error}"
        raise SystemExit(message) from error


if __name__ == "__main__":
    main()
