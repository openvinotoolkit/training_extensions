# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from fnmatch import fnmatchcase
from importlib import resources

import pytest

from app.services.demo_files_service import LICENSES_REQUIRING_ATTRIBUTION
from app.services.model_manifest_service import ModelManifestService
from app.supported_models.attributions import ModelAttribution, get_model_attribution, list_model_attributions
from app.supported_models.timm.manifest_provider import model_name_to_id


def _all_model_licenses() -> dict[str, str]:
    """License name of every model architecture: static model manifests and timm catalog."""
    licenses = {manifest.id: manifest.license.name for manifest in ModelManifestService.get_model_manifests().values()}
    snapshot = json.loads(
        resources.files("app.supported_models").joinpath("timm_catalog_snapshot.json").read_text(encoding="utf-8")
    )
    licenses.update({model_name_to_id(entry["model_name"]): entry["license"] for entry in snapshot["backbones"]})
    return licenses


class TestModelAttributions:
    def test_attributions_are_valid(self) -> None:
        attributions = list_model_attributions()

        assert attributions
        for attribution in attributions:
            assert attribution.models
            assert attribution.name.strip()
            assert attribution.creators.strip()
            assert attribution.source.startswith("https://")
            assert attribution.copyright is None or attribution.copyright.strip()

    def test_every_model_requiring_attribution_has_one(self) -> None:
        """Every model under a license requiring the retention of the original notices (MIT, BSD-3-Clause,
        CC BY-NC 4.0, ...) must have an attribution in `app/supported_models/licenses/attributions.yaml`."""
        missing = sorted(
            model_id
            for model_id, license_name in _all_model_licenses().items()
            if license_name in LICENSES_REQUIRING_ATTRIBUTION and get_model_attribution(model_id) is None
        )

        assert not missing, f"Models without attribution: {missing}"

    def test_no_model_matches_multiple_attributions(self) -> None:
        attributions = list_model_attributions()
        ambiguous = {
            model_id: names
            for model_id in _all_model_licenses()
            if len(names := [a.name for a in attributions if a.matches(model_id)]) > 1
        }

        assert not ambiguous, f"Models matching multiple attributions: {ambiguous}"

    def test_every_pattern_matches_a_model(self) -> None:
        """Guard against stale attributions, e.g. after a model is renamed or removed."""
        model_ids = list(_all_model_licenses())
        unused = [
            (attribution.name, pattern)
            for attribution in list_model_attributions()
            for pattern in attribution.models
            if not any(fnmatchcase(model_id, pattern) for model_id in model_ids)
        ]

        assert not unused, f"Attribution patterns not matching any model: {unused}"

    @pytest.mark.parametrize(
        "model_id, expected_name, expected_copyright",
        [
            ("image-classification-timm-resnet50.tv_in1k", "torchvision", "Copyright (c) Soumith Chintala 2016,"),
            ("image-classification-timm-swin_tiny_patch4_window7_224.ms_in1k", "Swin", "Microsoft Corporation"),
            ("image-classification-timm-hiera_tiny_224.mae", "Hiera", "Meta Platforms"),
            ("instance-segmentation-eomt-large-640", "EoMT", "Mobile Perception Systems Lab at TU/e"),
            ("image-classification-convnextv2-atto", "ConvNeXt V2", "Meta Platforms"),
        ],
    )
    def test_get_model_attribution(self, model_id: str, expected_name: str, expected_copyright: str) -> None:
        attribution = get_model_attribution(model_id)

        assert attribution is not None
        assert expected_name in attribution.name
        assert attribution.copyright is not None
        assert expected_copyright in attribution.copyright

    @pytest.mark.parametrize(
        "model_id",
        [
            "image-classification-timm-hiera_small_abswin_256.sbb2_e200_in12k",  # trained for timm, not by Meta
            "image-classification-timm-swinv2_cr_tiny_ns_224.sw_in1k",  # trained for timm, not by Microsoft
            "unknown-model",
        ],
    )
    def test_no_attribution(self, model_id: str) -> None:
        assert get_model_attribution(model_id) is None

    def test_matches(self) -> None:
        attribution = ModelAttribution(
            name="Foo", models=("model-foo-*",), creators="Foo Inc.", copyright=None, source="https://example.com"
        )

        assert attribution.matches("model-foo-large")
        assert not attribution.matches("model-bar-large")
