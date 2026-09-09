# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from app.models import ModelManifest, TaskType
from app.models.model_manifest import PretrainedWeights
from app.services.base_weights_service import BaseWeightsService
from app.services.model_manifest_service import ManifestNotFoundException, ModelManifestService

DETECTION_MODEL_MANIFEST_ID = "object-detection-atss-mobilenet-v2"
DETECTION_WEIGHTS_FILENAME = "mobilenet_v2-atss.pth"
CLASSIFICATION_MODEL_MANIFEST_ID = "image-classification-vit-tiny"
CACHE_FILENAME_OVERRIDE = "custom-cache-name.pth"
WEIGHTS_CONTENT = b"dummy pretrained weights content"


@pytest.fixture()
def fxt_pretrained_weights_dir(tmp_path: Path) -> Generator[Path]:
    """Set up a temporary data directory for tests."""
    pretrained_weights_dir = tmp_path / "pretrained_weights"
    pretrained_weights_dir.mkdir(parents=True, exist_ok=True)
    yield pretrained_weights_dir


@pytest.fixture
def fxt_base_weights_service(fxt_pretrained_weights_dir: Path) -> BaseWeightsService:
    """Create a BaseWeightsService instance for testing."""
    return BaseWeightsService(fxt_pretrained_weights_dir.parent)


@pytest.fixture
def fxt_manifest_with_cache_filename() -> ModelManifest:
    """Manifest whose url/mirror_url filenames differ from its explicit cache_filename."""
    manifest = ModelManifestService.get_model_manifest_by_id(DETECTION_MODEL_MANIFEST_ID)
    pretrained_weights = cast(PretrainedWeights, manifest.pretrained_weights)
    new_weights = pretrained_weights.model_copy(
        update={
            "url": "https://example.com/some/remote-name.pth",
            "mirror_url": "https://example.com/mirror/other-name.pth",
            "cache_filename": CACHE_FILENAME_OVERRIDE,
            "sha_sum": hashlib.sha256(WEIGHTS_CONTENT).hexdigest(),
        }
    )
    return manifest.model_copy(update={"pretrained_weights": new_weights})


class TestBaseWeightsService:
    """Test cases for the BaseWeightsService class."""

    def test_get_remote_weights_path_success(self, fxt_base_weights_service):
        """Test successful retrieval of remote weight's path."""
        result = fxt_base_weights_service.get_remote_weights_path(
            task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID
        )
        assert result == (
            "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions"
            "/models/object_detection/v2/mobilenet_v2-atss.pth"
        )

    def test_get_remote_weights_path_model_not_found(self, fxt_base_weights_service):
        """Test error when model manifest is not found."""
        with pytest.raises(ManifestNotFoundException, match="Model architecture with ID 'unknown_model' not found."):
            fxt_base_weights_service.get_remote_weights_path(task=TaskType.DETECTION, model_manifest_id="unknown_model")

    def test_get_remote_weights_path_task_mismatch(self, fxt_base_weights_service):
        """Test error when task doesn't match manifest."""
        with pytest.raises(ValueError, match="Task mismatch: expected 'classification', got 'detection'"):
            fxt_base_weights_service.get_remote_weights_path(
                task=TaskType.CLASSIFICATION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID
            )

    def test_get_local_weights_path_cached_valid(self, fxt_base_weights_service):
        """Test getting local weights path when file exists and is valid."""
        result = fxt_base_weights_service.get_local_weights_path(
            task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
        )

        expected_path = fxt_base_weights_service.pretrained_weights_dir / "detection" / DETECTION_WEIGHTS_FILENAME
        assert result == expected_path

    def test_get_local_weights_path_cached_invalid_redownload(self, fxt_base_weights_service):
        """Test redownloading when cached file fails integrity check."""
        # Download the first valid file
        result = fxt_base_weights_service.get_local_weights_path(
            task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
        )
        # Mock that the existing file is invalid
        with patch.object(fxt_base_weights_service, "_verify_file_integrity", side_effect=[False, True]):
            result = fxt_base_weights_service.get_local_weights_path(
                task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
            )

        expected_path = fxt_base_weights_service.pretrained_weights_dir / "detection" / DETECTION_WEIGHTS_FILENAME
        assert result == expected_path

    def test_get_local_weights_path_uses_cache_filename(
        self, fxt_base_weights_service, fxt_manifest_with_cache_filename
    ):
        """Test that cache_filename, not the url's filename, is used to key the local cache."""
        with (
            patch.object(
                ModelManifestService, "get_model_manifest_by_id", return_value=fxt_manifest_with_cache_filename
            ),
            patch.object(
                fxt_base_weights_service,
                "_download_weights",
                side_effect=lambda urls, local_path, sha_sum: local_path.write_bytes(WEIGHTS_CONTENT),
            ),
        ):
            result = fxt_base_weights_service.get_local_weights_path(
                task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
            )

        expected_path = fxt_base_weights_service.pretrained_weights_dir / "detection" / CACHE_FILENAME_OVERRIDE
        assert result == expected_path
        assert result.exists()

    def test_remove_local_weights_uses_cache_filename(self, fxt_base_weights_service, fxt_manifest_with_cache_filename):
        """Test that remove_local_weights deletes the file cached under cache_filename, not a url-derived name."""
        with (
            patch.object(
                ModelManifestService, "get_model_manifest_by_id", return_value=fxt_manifest_with_cache_filename
            ),
            patch.object(
                fxt_base_weights_service,
                "_download_weights",
                side_effect=lambda urls, local_path, sha_sum: local_path.write_bytes(WEIGHTS_CONTENT),
            ),
        ):
            result = fxt_base_weights_service.get_local_weights_path(
                task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
            )
            assert result.exists()

            is_deleted = fxt_base_weights_service.remove_local_weights(
                task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID
            )

        assert is_deleted
        assert not result.exists()

    def test_get_local_weights_path_download_required(self, fxt_base_weights_service):
        """Test redownloading when cached file does not exist."""
        with patch.object(Path, "exists", return_value=False):
            result = fxt_base_weights_service.get_local_weights_path(
                task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
            )

        expected_path = fxt_base_weights_service.pretrained_weights_dir / "detection" / DETECTION_WEIGHTS_FILENAME
        assert result == expected_path

    def test_get_local_weights_path_no_download_not_allowed(self, fxt_base_weights_service):
        """Test error when weights don't exist locally and download is not allowed."""
        with (
            patch.object(fxt_base_weights_service, "_verify_file_integrity", return_value=False),
            pytest.raises(
                FileNotFoundError,
                match=f"Weights not found locally for model {DETECTION_MODEL_MANIFEST_ID} and download is disabled",
            ),
        ):
            _ = fxt_base_weights_service.get_local_weights_path(
                task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=False
            )

    def test_remove_local_weights_success(self, fxt_base_weights_service):
        """Test successful removal of local weights."""
        result = fxt_base_weights_service.get_local_weights_path(
            task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
        )
        assert result.exists()
        is_deleted = fxt_base_weights_service.remove_local_weights(
            task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID
        )
        assert is_deleted

    def test_remove_local_weights_not_found(self, fxt_base_weights_service):
        """Test removing weights when file doesn't exist."""
        is_deleted = fxt_base_weights_service.remove_local_weights(
            task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID
        )
        assert not is_deleted

    def test_remove_all_local_weights(self, fxt_base_weights_service):
        """Test removing all cached weights."""
        detection = fxt_base_weights_service.get_local_weights_path(
            task=TaskType.DETECTION, model_manifest_id=DETECTION_MODEL_MANIFEST_ID, allow_download=True
        )
        classification = fxt_base_weights_service.get_local_weights_path(
            task=TaskType.CLASSIFICATION, model_manifest_id=CLASSIFICATION_MODEL_MANIFEST_ID, allow_download=True
        )
        assert detection.exists()
        assert classification.exists()
        count_removed = fxt_base_weights_service.remove_all_local_weights()
        assert count_removed == 2
        assert not detection.exists()
        assert not classification.exists()
