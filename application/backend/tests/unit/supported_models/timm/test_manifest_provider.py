# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType
from unittest.mock import patch

import pytest

from app.models.model_manifest import ModelManifestDeprecationStatus, WeightsSource
from app.models.task import TaskType
from app.supported_models.timm import catalog, manifest_provider
from app.supported_models.timm.manifest_provider import TimmManifestProvider, id_to_model_name, model_name_to_id

_FAKE_ENTRY = {
    "model_name": "resnet18.a1_in1k",
    "family": "resnet",
    "version": "resnet18",
    "pretrained": "a1_in1k",
    "input_size": [3, 224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "default_lr": 0.01,
    "default_weight_decay": 0.001,
    "imagenet_top1_accuracy": 70.0,
    "trainable_parameters": 11.7,
    "license": "apache-2.0",
    "gigaflops": 1.8,
}
_FAKE_SNAPSHOT = MappingProxyType({_FAKE_ENTRY["model_name"]: _FAKE_ENTRY})


@pytest.fixture(autouse=True)
def _fake_snapshot():
    """Replace the cached snapshot with a small, deterministic fixture."""
    catalog._snapshot.cache_clear()
    with patch.object(manifest_provider, "_snapshot", return_value=_FAKE_SNAPSHOT):
        yield


class TestModelIdConversion:
    def test_model_name_to_id_adds_prefix(self) -> None:
        assert model_name_to_id("resnet18.a1_in1k") == "image-classification-timm-resnet18.a1_in1k"

    def test_id_to_model_name_strips_prefix(self) -> None:
        assert id_to_model_name("image-classification-timm-resnet18.a1_in1k") == "resnet18.a1_in1k"

    def test_round_trip(self) -> None:
        model_name = "vit_base_patch16_224.augreg_in21k_ft_in1k"
        assert id_to_model_name(model_name_to_id(model_name)) == model_name

    def test_id_to_model_name_without_prefix_is_unchanged(self) -> None:
        assert id_to_model_name("not-a-timm-id") == "not-a-timm-id"


class TestTimmManifestProvider:
    def test_recognizes_timm_prefixed_id(self) -> None:
        assert TimmManifestProvider.is_timm_id("image-classification-timm-resnet18.a1_in1k") is True

    def test_rejects_non_timm_id(self) -> None:
        assert TimmManifestProvider.is_timm_id("image-classification-yolo-v8") is False

    def test_build_manifest_maps_snapshot_fields(self) -> None:
        manifest = TimmManifestProvider.build_manifest("resnet18.a1_in1k")

        assert manifest.id == model_name_to_id("resnet18.a1_in1k")
        assert manifest.name == "resnet18.a1_in1k"
        assert manifest.license == "apache-2.0"
        assert manifest.timm_metadata is not None
        assert manifest.timm_metadata.family == "resnet"
        assert manifest.timm_metadata.variant == "resnet18"
        assert manifest.timm_metadata.pretrained_tag == "a1_in1k"
        assert manifest.task == TaskType.CLASSIFICATION
        assert manifest.pretrained_weights.source == WeightsSource.TIMM
        assert manifest.support_status == ModelManifestDeprecationStatus.ACTIVE
        assert manifest.stats.gigaflops == 1.8
        assert manifest.stats.trainable_parameters == 11.7
        assert manifest.stats.benchmark_metrics.imagenet_top1_accuracy == 70.0
        assert manifest.hyperparameters.training.input_size_width == 224
        assert manifest.hyperparameters.training.input_size_height == 224
        assert manifest.hyperparameters.training.learning_rate == 0.01
        assert manifest.hyperparameters.training.weight_decay == 0.001

    def test_build_manifest_missing_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            TimmManifestProvider.build_manifest("unknown-model")

    def test_get_preprocessing_maps_snapshot_fields(self) -> None:
        preprocessing = TimmManifestProvider.get_preprocessing("resnet18.a1_in1k")

        assert preprocessing == {
            "input_size": (224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }
