# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pretrained-weight loader mixins."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.models.classification.utils.pretrained_weights import (
    PytorchcvWeightsLoader,
    TimmWeightsLoader,
    TorchvisionWeightsLoader,
    VisionTransformerWeightsLoader,
)


class _BackboneWeightsHarness:
    """Concrete attributes satisfying `_SupportsBackboneWeights` for testing."""

    def __init__(self, model_name: str) -> None:
        self.model = MagicMock()
        self.model_name = model_name


class _PytorchcvHarness(_BackboneWeightsHarness, PytorchcvWeightsLoader):
    """Testable `PytorchcvWeightsLoader` with concrete `model`/`model_name`."""


class _TorchvisionHarness(_BackboneWeightsHarness, TorchvisionWeightsLoader):
    """Testable `TorchvisionWeightsLoader` with concrete `model`/`model_name`."""


class _TimmHarness(_BackboneWeightsHarness, TimmWeightsLoader):
    """Testable `TimmWeightsLoader` with concrete `model`/`model_name`."""


class _VisionTransformerHarness(VisionTransformerWeightsLoader):
    """Concrete attributes satisfying `_SupportsViTBackboneWeights` for testing."""

    def __init__(self, model_name: str, pretrained_urls: dict[str, str] | None = None) -> None:
        self.model = MagicMock()
        self.model_name = model_name
        self.pretrained_urls = pretrained_urls or {}


class TestPytorchcvWeightsLoader:
    def test_load_pretrained_no_weights_downloads(self, tmp_path: Path):
        """When no local weights are given, falls back to pytorchcv's download_model."""
        loader = _PytorchcvHarness(model_name="efficientnet_b0")
        cache_dir = str(tmp_path / "cache")

        with (
            patch("pytorchcv.models.common.model_store.download_model") as mock_download,
            patch.dict("os.environ", {"PRETRAINED_WEIGHTS_CACHE_DIR": cache_dir}),
        ):
            loader.load_pretrained(weights=None)

        mock_download.assert_called_once()
        _, kwargs = mock_download.call_args
        assert kwargs["local_model_store_dir_path"] == cache_dir
        assert kwargs["model_name"] == "efficientnet_b0"

    def test_load_pretrained_extracts_zip_before_download(self, tmp_path: Path):
        """A cached .zip archive is extracted in place before delegating to download_model."""
        inner_name = "efficientnet_b0-0752-0e386130.pth"
        zip_path = tmp_path / f"{inner_name}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(inner_name, b"fake-checkpoint-bytes")

        loader = _PytorchcvHarness(model_name="efficientnet_b0")

        with patch("pytorchcv.models.common.model_store.download_model") as mock_download:
            loader.load_pretrained(weights=zip_path)

        extracted_path = tmp_path / inner_name
        assert extracted_path.exists()
        assert extracted_path.read_bytes() == b"fake-checkpoint-bytes"

        mock_download.assert_called_once()
        _, kwargs = mock_download.call_args
        assert kwargs["local_model_store_dir_path"] == str(tmp_path)

    def test_load_pretrained_non_zip_weights_skips_extraction(self, tmp_path: Path):
        """A non-.zip weights path is passed through untouched (as a cache-dir hint)."""
        pth_path = tmp_path / "efficientnet_b0-0752-0e386130.pth"
        pth_path.write_bytes(b"already-extracted")

        loader = _PytorchcvHarness(model_name="efficientnet_b0")

        with patch("pytorchcv.models.common.model_store.download_model") as mock_download:
            loader.load_pretrained(weights=pth_path)

        # File should be untouched, still present, not renamed/removed.
        assert pth_path.exists()
        mock_download.assert_called_once()
        _, kwargs = mock_download.call_args
        assert kwargs["local_model_store_dir_path"] == str(tmp_path)


class TestTorchvisionWeightsLoader:
    def test_load_pretrained_uses_local_checkpoint_when_present(self, tmp_path: Path):
        weights = tmp_path / "weights.pth"
        weights.write_bytes(b"data")

        loader = _TorchvisionHarness(model_name="efficientnet_b0")

        with patch(
            "getitune.backend.lightning.models.classification.utils.pretrained_weights.load_checkpoint"
        ) as mock_load_ckpt:
            loader.load_pretrained(weights=weights)

        mock_load_ckpt.assert_called_once_with(loader.model.backbone, str(weights))

    def test_load_pretrained_falls_back_to_torchvision_default(self, tmp_path: Path):
        missing_weights = tmp_path / "does-not-exist.pth"

        loader = _TorchvisionHarness(model_name="efficientnet_b0")

        with (
            patch("torchvision.models.get_model") as mock_get_model,
            patch("torchvision.models.get_model_weights") as mock_get_weights,
        ):
            mock_get_weights.return_value.verify.return_value = "DEFAULT"
            loader.load_pretrained(weights=missing_weights)

        mock_get_model.assert_called_once()


class TestTimmWeightsLoader:
    def test_load_pretrained_uses_local_checkpoint_when_present(self, tmp_path: Path):
        weights = tmp_path / "weights.pth"
        weights.write_bytes(b"data")

        loader = _TimmHarness(model_name="vit_tiny")

        with patch(
            "getitune.backend.lightning.models.classification.utils.pretrained_weights.load_checkpoint"
        ) as mock_load_ckpt:
            loader.load_pretrained(weights=weights)

        mock_load_ckpt.assert_called_once_with(loader.model.backbone.model, str(weights))

    def test_load_pretrained_falls_back_to_timm_default(self):
        loader = _TimmHarness(model_name="vit_tiny")

        with patch("timm.models.load_pretrained") as mock_timm_load:
            loader.load_pretrained(weights=None)

        mock_timm_load.assert_called_once()


class TestVisionTransformerWeightsLoader:
    def test_load_pretrained_uses_local_checkpoint_when_present(self, tmp_path: Path):
        weights = tmp_path / "vit_weights.pth"
        weights.write_bytes(b"data")

        loader = _VisionTransformerHarness(model_name="vit_tiny")

        loader.load_pretrained(weights=weights)

        loader.model.backbone.load_checkpoint.assert_called_once_with(checkpoint_path=weights)

    def test_load_pretrained_warns_when_model_name_unknown(self):
        loader = _VisionTransformerHarness(model_name="vit_tiny")

        with pytest.warns(UserWarning, match="random weights"):
            loader.load_pretrained(weights=None)

        loader.model.backbone.load_checkpoint.assert_not_called()

    def test_load_pretrained_downloads_when_url_known(self, tmp_path: Path):
        loader = _VisionTransformerHarness(
            model_name="vit_tiny",
            pretrained_urls={"vit_tiny": "https://example.com/vit_tiny.pth"},
        )

        with (
            patch(
                "getitune.backend.lightning.models.classification.utils.pretrained_weights.download_url_to_file"
            ) as mock_download,
            patch.dict("os.environ", {"PRETRAINED_WEIGHTS_CACHE_DIR": str(tmp_path)}),
        ):
            loader.load_pretrained(weights=None)

        mock_download.assert_called_once_with(
            "https://example.com/vit_tiny.pth", str(tmp_path / "vit_tiny.pth"), "", progress=True
        )
        loader.model.backbone.load_checkpoint.assert_called_once()
