# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export a getitune model to ONNX or OpenVINO with the requested precision."""

from __future__ import annotations

import argparse
import logging
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.hub import download_url_to_file

from getitune.backend.lightning.models.classification.heads.vision_transformer_head import VisionTransformerClsHead
from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision

logger = logging.getLogger(__name__)

_IMAGENET_NUM_CLASSES = 1000
_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
_DINOV2_HEAD_URLS: dict[str, str] = {
    "dinov2-small": f"{_DINOV2_BASE_URL}/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth",
    "dinov2-base": f"{_DINOV2_BASE_URL}/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth",
    "dinov2-large": f"{_DINOV2_BASE_URL}/dinov2_vitl14/dinov2_vitl14_reg4_linear_head.pth",
    "dinov2-giant": f"{_DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_reg4_linear_head.pth",
}


def _dinov2_backbone_forward(self: nn.Module, x: torch.Tensor, out_type: str = "cls_token") -> tuple:
    """Backbone forward returning ``concat(cls_token, mean(patch_tokens))`` for the default call.

    Mirrors ``create_linear_input`` of facebookresearch/dinov2 with ``n_last_blocks=1`` and ``use_avgpool=True``.
    Other ``out_type`` values (e.g. ``"raw"``) fall through to the original implementation.
    """
    original_forward = type(self).forward
    if out_type != "cls_token":
        return original_forward(self, x, out_type=out_type)
    tokens = original_forward(self, x, out_type="raw")[-1]  # [B, N, D], already normalized
    cls_token = tokens[:, 0]
    patch_mean = tokens[:, self.num_prefix_tokens :].mean(dim=1)  # skip cls + register tokens
    return (torch.cat([cls_token, patch_mean], dim=-1),)


def _apply_dinov2_linear_head(engine: Any, head_weights: Path | None, cache_dir: Path) -> None:  # noqa: ANN401
    """Patch the model instance to use the official facebookresearch/dinov2 linear head (export only).

    Args:
        engine: Engine holding a DINOv2 ``VisionTransformer*Cls`` model.
        head_weights: Local ``*_linear_head.pth`` file; downloaded into ``cache_dir`` if None.
        cache_dir: Directory where the downloaded head is stored.

    Raises:
        FileNotFoundError: If the given head weights file does not exist.
        ValueError: If the model is not a multi-class DINOv2 classifier or the head shape does not match the model.
    """
    lightning_model = engine.model
    model_name = getattr(lightning_model, "model_name", "")
    if model_name not in _DINOV2_HEAD_URLS:
        message = f"--dinov2-linear-head requires a DINOv2 model, got {model_name!r}"
        raise ValueError(message)

    classifier = lightning_model.model
    head = getattr(classifier, "head", None)
    if not isinstance(head, VisionTransformerClsHead) or not hasattr(head.layers, "head"):
        message = (
            "--dinov2-linear-head supports only multi-class classification (MULTI_CLASS_CLS) DINOv2 models; "
            f"got head {type(head).__name__!r} for task {getattr(lightning_model, 'task', None)!s}"
        )
        raise ValueError(message)

    if head_weights is None:
        url = _DINOV2_HEAD_URLS[model_name]
        head_weights = cache_dir / Path(url).name
        if not head_weights.exists():
            download_url_to_file(url, str(head_weights), progress=True)
    if not head_weights.is_file():
        message = f"DINOv2 linear head weights not found: {head_weights}"
        raise FileNotFoundError(message)
    state = torch.load(head_weights, map_location="cpu", weights_only=True)

    backbone = classifier.backbone
    embed_dim = backbone.embed_dim
    out_features, in_features = state["weight"].shape
    if in_features != 2 * embed_dim:
        message = (
            f"Head expects {in_features} input features but backbone provides 2*{embed_dim}; "
            f"use the *_linear_head.pth matching {model_name!r}"
        )
        raise ValueError(message)
    if out_features != _IMAGENET_NUM_CLASSES:
        message = (
            f"Head weights have {out_features} outputs, expected {_IMAGENET_NUM_CLASSES}; "
            "--dinov2-linear-head requires the official ImageNet-1k *_linear_head.pth"
        )
        raise ValueError(message)
    if lightning_model.num_classes != _IMAGENET_NUM_CLASSES:
        message = (
            f"Official head has {_IMAGENET_NUM_CLASSES} classes but the model has {lightning_model.num_classes}; "
            "use a dataset with the 1000 ImageNet classes (in ImageNet order)"
        )
        raise ValueError(message)

    # 1) Backbone: default call returns concat(cls, mean(patch tokens)) instead of cls token only.
    backbone.forward = types.MethodType(_dinov2_backbone_forward, backbone)

    # 2) Head: replace the final Linear with the official one.
    linear = nn.Linear(in_features, out_features)
    linear.load_state_dict(state)
    linear.eval()
    head.layers.head = linear
    head.in_channels = in_features
    logger.info("Applied DINOv2 linear head from %s", head_weights)


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

    if args.dinov2_linear_head is not None and args.checkpoint is not None:
        message = "--dinov2-linear-head cannot be combined with --checkpoint"
        raise ValueError(message)

    engine = create_engine(
        model=args.model,
        data=str(args.data_root),
        work_dir=str(args.work_dir),
        device=args.device,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        task=args.task,
    )

    if args.dinov2_linear_head is not None:
        head_path = None if args.dinov2_linear_head == "auto" else Path(args.dinov2_linear_head)
        _apply_dinov2_linear_head(engine, head_path, args.work_dir)

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
    parser.add_argument(
        "--dinov2-linear-head",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Initialize the head of a DINOv2 model with pretrained weights."
            "You can either provide the path to *_linear_head.pth file, "
            "or omit the value to download the official facebookresearch/dinov2 weights "
            "pretrained on ImageNet-1K."
        ),
    )
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
