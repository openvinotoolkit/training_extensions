"""Export pretrained RF-DETR COCO models with their original 91 output slots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rfdetr.assets.coco_classes import COCO_CLASSES
from torch.hub import download_url_to_file

from getitune.backend.lightning.models.detection.rfdetr import RFDETR
from getitune.backend.lightning.models.instance_segmentation.rfdetr_inst import RFDETRInst
from getitune.types.export import ExportFormat
from getitune.types.label import LabelInfo
from getitune.types.precision import Precision

COCO_SLOT_COUNT = 91
SUPPORTED_MODELS = (
    "rfdetr_nano",
    "rfdetr_small",
    "rfdetr_medium",
    "rfdetr_large",
    "rfdetr_seg_n",
    "rfdetr_seg_s",
    "rfdetr_seg_m",
    "rfdetr_seg_l",
    "rfdetr_seg_xl",
    "rfdetr_seg_2xl",
)


def build_coco_label_info() -> LabelInfo:
    """Return metadata indexed by the original COCO category ID."""
    label_names = [COCO_CLASSES.get(slot, f"coco_unused_slot_{slot}") for slot in range(COCO_SLOT_COUNT)]
    label_ids = [str(slot) for slot in range(COCO_SLOT_COUNT)]
    return LabelInfo(label_names=label_names, label_groups=[label_names], label_ids=label_ids)


def assert_coco_classifier_heads(model: RFDETR | RFDETRInst) -> None:
    """Fail if pretrained loading did not retain all 91 classifier rows."""
    lwdetr = model.model.lwdetr
    classifier_heads = {"decoder": lwdetr.class_embed}
    if lwdetr.two_stage:
        classifier_heads.update(
            {f"encoder_{index}": head for index, head in enumerate(lwdetr.transformer.enc_out_class_embed)}
        )

    for name, head in classifier_heads.items():
        if head.bias is None or head.weight.shape[0] != COCO_SLOT_COUNT or head.bias.shape[0] != COCO_SLOT_COUNT:
            rows = head.weight.shape[0]
            message = f"{model.model_name} {name} classifier has {rows} rows; expected {COCO_SLOT_COUNT}"
            raise RuntimeError(message)


def resolve_pretrained_weights(model_type: type[RFDETR] | type[RFDETRInst], model_name: str) -> Path:
    """Download each variant to a distinct cache path, avoiding shared URL basenames."""
    url = model_type.pretrained_urls[model_name]
    cache_dir = Path(os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / f"getitune_rfdetr_coco91_{model_name}.pth"
    if not weights_path.exists():
        download_url_to_file(url, str(weights_path), progress=os.isatty(0))
    return weights_path


def configure_export_labels(model: RFDETR | RFDETRInst) -> None:
    """Attach the raw-slot metadata without resizing the already-loaded model."""
    raw_label_info = build_coco_label_info()
    model._label_info = raw_label_info  # noqa: SLF001
    parameters = model._export_parameters  # noqa: SLF001
    metadata = parameters.to_metadata()

    expected_names = raw_label_info.label_names
    expected_ids = raw_label_info.label_ids
    if isinstance(model, RFDETRInst):
        expected_names = ["getitune_empty_lbl", *expected_names]
        expected_ids = ["None", *expected_ids]

    if parameters.label_info.label_names != expected_names or parameters.label_info.label_ids != expected_ids:
        message = f"{model.model_name} export label metadata does not preserve all COCO output slots"
        raise RuntimeError(message)
    if metadata[("model_info", "confidence_threshold")] != "0.5":
        message = f"{model.model_name} export confidence threshold must be 0.5"
        raise RuntimeError(message)
    if ("model_info", "iou_threshold") in metadata or metadata[("model_info", "nms_execute")] != "False":
        message = f"{model.model_name} export metadata must disable NMS"
        raise RuntimeError(message)


def export_model(model_name: str, output_dir: Path) -> Path:
    """Load and export one official pretrained RF-DETR variant."""
    model_type = RFDETRInst if model_name.startswith("rfdetr_seg_") else RFDETR
    weights_path = resolve_pretrained_weights(model_type, model_name)
    model = model_type(
        model_name=model_name,
        label_info=LabelInfo.from_num_classes(COCO_SLOT_COUNT - 1),
        pretrained=True,
        pretrained_weights=weights_path,
    )
    assert_coco_classifier_heads(model)
    configure_export_labels(model)
    exported = model.export(output_dir, model_name, ExportFormat.OPENVINO, Precision.FP16)
    print(f"Exported {model_name}: {exported}")
    return exported


def build_parser() -> argparse.ArgumentParser:
    """Build the one-off export CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help="Variants to export (default: all supported detection and segmentation variants)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./rfdetr-coco91-openvino"))
    return parser


def main() -> None:
    """Export the selected official pretrained variants."""
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        export_model(model_name, args.output_dir)


if __name__ == "__main__":
    main()
