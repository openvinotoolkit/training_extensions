#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from importlib import resources
from unittest.mock import mock_open, patch

import pytest

from app.models import TaskType
from app.models.model_manifest import (
    BenchmarkMetrics,
    Capabilities,
    DirectLinkPretrainedWeights,
    ModelManifest,
    ModelManifestDeprecationStatus,
    ModelStats,
)
from app.models.training_configuration import (
    AlgoLevelDatasetPreparationParameters,
    AlgoLevelParameters,
    AlgoLevelTrainingParameters,
)
from app.models.training_configuration.training import EarlyStopping
from app.services.model_manifest_service import ModelManifestService
from app.supported_models import manifests

BASE_MANIFEST_PATH = str(resources.files(manifests).joinpath("base.yaml"))
TEST_PATH = pathlib.Path(os.path.dirname(__file__))
DUMMY_BASE_MANIFEST_PATH = os.path.join(TEST_PATH, "dummy_base_model_manifest.yaml")
DUMMY_MANIFEST_PATH = os.path.join(TEST_PATH, "dummy_model_manifest.yaml")


@pytest.fixture
def fxt_dummy_model_stats():
    yield ModelStats(
        gigaflops=0.39,
        trainable_parameters=5288548,
        benchmark_metrics=BenchmarkMetrics(
            imagenet_top1_accuracy=76.2,
            imagenet_top5_accuracy=95.3,
        ),
    )


@pytest.fixture
def fxt_dummy_pretrained_weights():
    yield DirectLinkPretrainedWeights(
        url="https://example.com/dummy_model_weights.pth",
        mirror_url="https://mirror.example.com/dummy_model_weights.pth",
        sha_sum="example_sha256_checksum",
    )


@pytest.fixture
def fxt_dummy_hyperparameters():
    yield AlgoLevelParameters(
        dataset_preparation=AlgoLevelDatasetPreparationParameters(),
        training=AlgoLevelTrainingParameters(
            max_epochs=101,
            learning_rate=0.05,
            early_stopping=EarlyStopping(patience=5),
            allowed_values_input_size=[128, 256, 512],
            input_size_width=512,
            input_size_height=256,
        ),
    )


@pytest.fixture
def fxt_dummy_model_manifest(fxt_dummy_model_stats, fxt_dummy_pretrained_weights, fxt_dummy_hyperparameters):
    yield ModelManifest(
        id="dummy_model_manifest",
        name="Dummy ModelManifest",
        pretrained_weights=fxt_dummy_pretrained_weights,
        description="Dummy manifest for test purposes only",
        stats=fxt_dummy_model_stats,
        support_status=ModelManifestDeprecationStatus.OBSOLETE,
        hyperparameters=fxt_dummy_hyperparameters,
        capabilities=Capabilities(xai=True, tiling=False),
        task=TaskType.CLASSIFICATION,
    )


class TestModelManifestService:
    def test_parse_manifest(self, fxt_dummy_model_manifest):
        model_manifest = ModelManifestService._parse_manifest(
            BASE_MANIFEST_PATH, DUMMY_BASE_MANIFEST_PATH, DUMMY_MANIFEST_PATH, relative=False
        )

        assert model_manifest == fxt_dummy_model_manifest

    @pytest.mark.parametrize("license_yaml", [None, "MIT"])
    def test_parse_manifest_with_relative_path(self, license_yaml):
        sources = ("base.yaml", "dummy_base_model_manifest.yaml", "dummy_model_manifest.yaml")
        expected_paths = [str(resources.files(manifests).joinpath(path)) for path in sources]

        # Create a more complete mock result with all required nested fields
        mock_yaml_result = {
            "id": "test",
            "name": "Test Model",
            "pretrained_weights": {
                "source": "direct_link",
                "url": "https://example.com/test_model_weights.pth",
                "mirror_url": "https://mirror.example.com/test_model_weights.pth",
                "sha_sum": "test_sha256_checksum",
            },
            "description": "Test",
            "task": "detection",
            "stats": {
                "gigaflops": 1.0,
                "trainable_parameters": 1000,
                "benchmark_metrics": {"coco_map_50_95": 54.0, "coco_map_50": 71.6},
            },
            "support_status": "active",
            "hyperparameters": {
                "dataset_preparation": {
                    "augmentation": {
                        "gaussian_blur": {"kernel_size": 5, "sigma": [0.1, 2.0], "probability": 0.8},
                        "tiling": {"enable_adaptive_tiling": True, "tile_size": 100, "tile_overlap": 0.3},
                    }
                },
                "training": {
                    "max_epochs": 100,
                    "learning_rate": 0.01,
                    "early_stopping": {"patience": 3},
                    "allowed_values_input_size": [128, 256, 512],
                    "input_size_width": 512,
                    "input_size_height": 256,
                },
                "evaluation": {"metric": None},
            },
            "capabilities": {
                "xai": True,
                "tiling": False,
            },
        }
        if license_yaml:
            mock_yaml_result["license"] = license_yaml

        with (
            patch("app.services.model_manifest_service.open", mock_open(), create=True) as mock_file,
            patch("app.services.model_manifest_service.yaml.safe_load") as mock_safe_load,
        ):
            # Each source file yields the same complete manifest; deep-merging identical
            # dicts produces an equivalent result.
            mock_safe_load.return_value = mock_yaml_result
            model_manifest = ModelManifestService._parse_manifest(*sources, relative=True)

            # Verify the relative sources were resolved to the expected absolute paths.
            opened_paths = [call.args[0] for call in mock_file.call_args_list]
            assert opened_paths == expected_paths
            assert model_manifest == ModelManifest(**mock_yaml_result)  # pyrefly: ignore[bad-argument-type]
            if not license_yaml:
                assert model_manifest.license == "Apache 2.0"

    def test_get_model_manifests(self) -> None:
        # test that the model manifests can be retrieved without errors
        model_manifests = ModelManifestService.get_model_manifests()

        assert len(model_manifests) > 0

    @pytest.mark.parametrize(
        "model_manifest_id, expected_task",
        [
            ("image-classification-efficientnet-v2-s", "classification"),
            ("object-detection-atss-mobilenet-v2", "detection"),
            ("instance-segmentation-mask-rcnn-efficientnet-b2", "instance_segmentation"),
        ],
    )
    def test_get_model_manifest_by_id(self, model_manifest_id, expected_task) -> None:
        model_manifest = ModelManifestService.get_model_manifest_by_id(model_manifest_id)

        assert model_manifest.id == model_manifest_id
        assert model_manifest.task == expected_task
