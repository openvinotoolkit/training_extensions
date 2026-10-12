from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.export_rfdetr_coco91 import (
    COCO_SLOT_COUNT,
    assert_coco_classifier_heads,
    build_coco_label_info,
    resolve_pretrained_weights,
)

from getitune.backend.lightning.models.detection.rfdetr import RFDETR


def test_coco_label_info_preserves_sparse_output_slots() -> None:
    label_info = build_coco_label_info()

    assert len(label_info.label_names) == COCO_SLOT_COUNT
    assert label_info.label_ids == [str(slot) for slot in range(COCO_SLOT_COUNT)]
    assert label_info.label_names[0] == "coco_unused_slot_0"
    assert label_info.label_names[12] == "coco_unused_slot_12"
    assert label_info.label_names[18] == "dog"
    assert label_info.label_names[47] == "cup"
    assert label_info.label_names[90] == "toothbrush"
    assert label_info.label_groups[0] is label_info.label_names


def test_classifier_head_validation_includes_decoder_and_encoder() -> None:
    model = SimpleNamespace(
        model_name="rfdetr_nano",
        model=SimpleNamespace(
            lwdetr=SimpleNamespace(
                class_embed=torch.nn.Linear(4, COCO_SLOT_COUNT),
                two_stage=True,
                transformer=SimpleNamespace(enc_out_class_embed=[torch.nn.Linear(4, COCO_SLOT_COUNT)]),
            ),
        ),
    )

    assert_coco_classifier_heads(model)

    model.model.lwdetr.transformer.enc_out_class_embed[0] = torch.nn.Linear(4, COCO_SLOT_COUNT - 1)
    with pytest.raises(RuntimeError, match="encoder_0 classifier has 90 rows"):
        assert_coco_classifier_heads(model)


def test_detection_variants_use_distinct_weight_cache_paths(monkeypatch, tmp_path) -> None:
    downloads = []

    def record_download(url: str, path: str, progress: bool) -> None:
        downloads.append((url, Path(path)))
        Path(path).touch()

    monkeypatch.setenv("PRETRAINED_WEIGHTS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("scripts.export_rfdetr_coco91.download_url_to_file", record_download)

    paths = [resolve_pretrained_weights(RFDETR, name) for name in ("rfdetr_nano", "rfdetr_small", "rfdetr_medium")]

    assert len(set(paths)) == 3
    assert [url for url, _ in downloads] == [
        RFDETR.pretrained_urls[name] for name in ("rfdetr_nano", "rfdetr_small", "rfdetr_medium")
    ]
