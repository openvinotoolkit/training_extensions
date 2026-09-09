# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Security regression tests for LightningEngine checkpoint loading.

Checkpoints must be loaded exclusively through PyTorch's safe deserialization
(``weights_only=True``). A malicious checkpoint embedding a ``__reduce__``
payload must never execute code while being loaded.
"""

import os
from pathlib import Path

import pytest
import torch

from getitune.backend.lightning.engine import LightningEngine
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.classification.multiclass_models import EfficientNetMulticlassCls
from getitune.types.label import LabelInfo


@pytest.fixture
def fxt_engine(tmp_path) -> LightningEngine:
    return LightningEngine(
        data="tests/assets/classification_cifar10",
        model="src/getitune/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml",
        work_dir=tmp_path,
        max_epochs=9,
    )


def test_load_model_checkpoint_rejects_malicious_pickle(tmp_path) -> None:
    """A checkpoint with a ``__reduce__`` payload is refused and executes no code."""
    marker = tmp_path / "rce_marker"
    payload_cmd = f"touch {marker}"
    checkpoint = tmp_path / "malicious.ckpt"

    class _Payload:
        def __reduce__(self) -> tuple:
            return (os.system, (payload_cmd,))

    torch.save({"state_dict": {}, "payload": _Payload()}, checkpoint)

    # The loader fails before touching the model, so a bare instance is enough here.
    engine = object.__new__(LightningEngine)
    with pytest.raises(RuntimeError, match="weights_only=True"):
        engine._load_model_checkpoint(checkpoint, map_location="cpu")

    assert not marker.exists(), "Malicious checkpoint must not execute code during load"


def test_load_model_checkpoint_rejects_arbitrary_pickled_objects(tmp_path, fxt_engine) -> None:
    """Pickled non-allowlisted objects (e.g. custom dataclasses in hyper_parameters) are refused."""
    checkpoint = tmp_path / "custom_objects.ckpt"
    torch.save(
        {
            "state_dict": fxt_engine.model.state_dict(),
            "hyper_parameters": {"label_info": LabelInfo(label_names=["a"], label_ids=["0"], label_groups=[["a"]])},
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="weights_only=True"):
        fxt_engine._load_model_checkpoint(checkpoint, map_location="cpu")


def test_load_model_checkpoint_loads_state_dict(tmp_path, fxt_engine) -> None:
    """A plain tensor state_dict checkpoint loads without executing or refusing anything."""
    checkpoint = tmp_path / "plain.ckpt"
    torch.save({"state_dict": fxt_engine.model.state_dict()}, checkpoint)

    ckpt = fxt_engine._load_model_checkpoint(checkpoint, map_location="cpu")

    assert "state_dict" in ckpt
    assert set(ckpt["state_dict"]) == set(fxt_engine.model.state_dict())


def test_load_model_checkpoint_loads_checkpoint_with_pathlib_path(tmp_path, fxt_engine) -> None:
    """Checkpoints saved by getitune <= 0.3.0 pickle `pretrained_weights` as a Path and must still load."""
    checkpoint = tmp_path / "path_hparams.ckpt"
    torch.save(
        {
            "state_dict": fxt_engine.model.state_dict(),
            "hyper_parameters": {"pretrained_weights": Path(tmp_path / "dummy_weights.pt")},
        },
        checkpoint,
    )

    ckpt = fxt_engine._load_model_checkpoint(checkpoint, map_location="cpu")

    assert ckpt["hyper_parameters"]["pretrained_weights"] == Path(tmp_path / "dummy_weights.pt")


def test_save_hyperparameters_excludes_pretrained_weights(tmp_path) -> None:
    """`pretrained_weights` init arg must not be pickled into saved checkpoints."""
    model = EfficientNetMulticlassCls(
        model_name="efficientnet_b0",
        label_info=3,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        pretrained=False,
        pretrained_weights=tmp_path / "dummy_weights.pt",
    )

    assert "pretrained_weights" not in model.hparams
