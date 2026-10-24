# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: standalone validator DataLoaders must follow device, not hardcode pin_memory=True."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from getitune.backend.ultralytics.validators.classification import (
    ClassificationValidator,
    MultiLabelClassificationValidator,
)
from getitune.backend.ultralytics.validators.detection import DetectionValidator
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator
from getitune.backend.ultralytics.validators.semantic_segmentation import SemanticSegmentationValidator
from getitune.backend.ultralytics.validators.yolo_detr import YoloDetrValidator

# Validators that fall through to GetiTuneValidatorMixin._build_adapter_dataloader
# share one DataLoader import; ClassificationValidator/MultiLabelClassificationValidator
# and SemanticSegmentationValidator override it with their own import.
_VALIDATOR_CASES = [
    pytest.param(DetectionValidator, "getitune.backend.ultralytics.validators.base.DataLoader", id="detection"),
    pytest.param(
        SegmentationValidator,
        "getitune.backend.ultralytics.validators.base.DataLoader",
        id="instance_segmentation",
    ),
    pytest.param(YoloDetrValidator, "getitune.backend.ultralytics.validators.base.DataLoader", id="yolo_detr"),
    pytest.param(
        ClassificationValidator,
        "getitune.backend.ultralytics.validators.classification.DataLoader",
        id="classification",
    ),
    pytest.param(
        MultiLabelClassificationValidator,
        "getitune.backend.ultralytics.validators.classification.DataLoader",
        id="multilabel_classification",
    ),
    pytest.param(
        SemanticSegmentationValidator,
        "getitune.backend.ultralytics.validators.semantic_segmentation.DataLoader",
        id="semantic_segmentation",
    ),
]


def _make_datamodule() -> SimpleNamespace:
    subset = MagicMock()
    return SimpleNamespace(
        test_subset=SimpleNamespace(subset_name="test", num_workers=0),
        val_subset=SimpleNamespace(subset_name="val", num_workers=0),
        subsets={"test": subset, "val": subset},
    )


class TestValidatorPinMemoryPolicy:
    """Standalone validator DataLoader pin_memory must follow device, across every validator."""

    @pytest.mark.parametrize(("validator_cls", "loader_patch_target"), _VALIDATOR_CASES)
    @pytest.mark.parametrize(
        ("device_type", "expected_pin_memory"),
        [
            ("cpu", False),
            ("cuda:0", True),
            ("xpu:0", True),
        ],
    )
    def test_build_adapter_dataloader_respects_device(
        self,
        device_type: str,
        expected_pin_memory: bool,
        validator_cls: type,
        loader_patch_target: str,
    ) -> None:
        validator = object.__new__(validator_cls)
        validator._datamodule = _make_datamodule()
        validator.device = torch.device(device_type)
        validator.args = SimpleNamespace(batch=4)

        with patch(loader_patch_target) as mock_dataloader_cls:
            validator._build_adapter_dataloader()

        _, kwargs = mock_dataloader_cls.call_args
        assert kwargs["pin_memory"] is expected_pin_memory

    @pytest.mark.parametrize(("validator_cls", "loader_patch_target"), _VALIDATOR_CASES)
    @pytest.mark.parametrize("num_workers", [0, 3])
    def test_build_adapter_dataloader_uses_test_workers(
        self, validator_cls: type, loader_patch_target: str, num_workers: int
    ) -> None:
        validator = object.__new__(validator_cls)
        validator._datamodule = _make_datamodule()
        validator._datamodule.test_subset.num_workers = num_workers
        validator.device = torch.device("cpu")
        validator.args = SimpleNamespace(batch=4)

        with patch(loader_patch_target) as mock_dataloader_cls:
            validator._build_adapter_dataloader()

        _, kwargs = mock_dataloader_cls.call_args
        assert kwargs["num_workers"] == num_workers
        assert kwargs["persistent_workers"] is (num_workers > 0)
        assert (kwargs["multiprocessing_context"] is not None) is (num_workers > 0)

    @pytest.mark.parametrize(("validator_cls", "loader_patch_target"), _VALIDATOR_CASES)
    @pytest.mark.parametrize("empty_test_subset", [False, True])
    def test_build_adapter_dataloader_falls_back_to_val_workers(
        self, validator_cls: type, loader_patch_target: str, empty_test_subset: bool
    ) -> None:
        validator = object.__new__(validator_cls)
        validator._datamodule = _make_datamodule()
        if empty_test_subset:
            validator._datamodule.subsets["test"] = []
        else:
            validator._datamodule.subsets.pop("test")
        validator._datamodule.val_subset.num_workers = 2
        validator.device = torch.device("cpu")
        validator.args = SimpleNamespace(batch=4)

        with patch(loader_patch_target) as mock_dataloader_cls:
            validator._build_adapter_dataloader()

        _, kwargs = mock_dataloader_cls.call_args
        assert kwargs["num_workers"] == 2
