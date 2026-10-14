# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.supported_models.timm import TimmCatalog, TimmManifestProvider, model_name_to_id


class TestModelArchitecturesEndpoint:
    """Test cases for the model architectures endpoint."""

    def test_get_all_model_architectures(self, fxt_client: TestClient):
        """Test getting all model architectures without filtering."""
        response = fxt_client.get("/api/model_architectures?task=detection")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "model_architectures" in data
        assert len(data["model_architectures"]) == 40

        # Verify structure of first detection model
        detection_model = next(
            arch for arch in data["model_architectures"] if arch["id"] == "object-detection-atss-mobilenet-v2"
        )

        assert detection_model["task"] == "detection"
        assert detection_model["name"] == "ATSS-MobileNet-V2"
        assert (
            detection_model["description"]
            == "ATSS (Adaptive Training Sample Selection) is an anchor-based object detection algorithm that introduces"
            " an adaptive strategy for selecting positive and negative samples during training. Instead of using"
            " fixed IoU thresholds, ATSS dynamically determines positive samples based on the statistical"
            " characteristics of object candidates for each ground truth. This improves training stability"
            " and enhances detection performance, especially for objects of varying sizes and aspect ratios."
        )
        assert detection_model["support_status"] == "active"

        # Verify capabilities structure
        assert "capabilities" in detection_model
        assert detection_model["capabilities"]["xai"] is True
        assert detection_model["capabilities"]["tiling"] is True

        # Verify stats structure
        assert "stats" in detection_model
        assert detection_model["stats"]["gigaflops"] == 20.6
        assert detection_model["stats"]["trainable_parameters"] == 3.9
        assert "benchmark_metrics" in detection_model["stats"]

        # Verify top picks
        assert "top_picks" in data
        top_picks = data["top_picks"]
        assert top_picks["balance"] == "object-detection-dfine-m"
        assert top_picks["speed"] == "object-detection-yolox-s"
        assert top_picks["accuracy"] == "object-detection-dfine-l"

    @pytest.mark.parametrize(
        "task_filter, total_models",
        [
            ("detection", 40),
            ("instance_segmentation", 25),
            ("classification", 22),
        ],
    )
    def test_get_model_architectures_various_tasks(self, fxt_client: TestClient, task_filter, total_models):
        """Test getting model architectures with various task filters."""
        response = fxt_client.get(f"/api/model_architectures?task={task_filter}")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "model_architectures" in data
        assert len(data["model_architectures"]) == total_models

        for model in data["model_architectures"]:
            assert model["task"].lower() == task_filter.lower()

    def test_get_model_architectures_nonexistent_task_filter(self, fxt_client: TestClient):
        """Test filtering by a task that doesn't exist returns 422."""
        response = fxt_client.get("/api/model_architectures?task=nonexistent")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_get_timm_families(self, fxt_client: TestClient):
        """Families list is non-empty, sorted, and de-duplicated."""
        response = fxt_client.get("/api/model_architectures/timm/families")
        assert response.status_code == status.HTTP_200_OK

        families = response.json()
        assert isinstance(families, list)
        assert len(families) > 0
        assert families == sorted(set(families))
        assert all(isinstance(f, str) for f in families)

    def test_get_timm_variants_for_known_family(self, fxt_client: TestClient):
        """Variants are returned for a family that exists in the catalog."""
        families = fxt_client.get("/api/model_architectures/timm/families").json()
        family = families[0]

        response = fxt_client.get(f"/api/model_architectures/timm/families/{family}/variants")
        assert response.status_code == status.HTTP_200_OK

        variants = response.json()
        assert isinstance(variants, list)
        assert len(variants) > 0
        assert variants == sorted(set(variants))

    def test_get_timm_variants_for_unknown_family_returns_empty(self, fxt_client: TestClient):
        """An unknown family yields an empty list rather than an error."""
        response = fxt_client.get("/api/model_architectures/timm/families/nonexistent-family/variants")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_get_timm_pretrained_tags_for_known_family_and_version(self, fxt_client: TestClient):
        """Pretrained tags are returned for a valid family/version pair."""
        families = fxt_client.get("/api/model_architectures/timm/families").json()
        family = families[0]
        variants = fxt_client.get(f"/api/model_architectures/timm/families/{family}/variants").json()
        variant = variants[0]

        response = fxt_client.get(f"/api/model_architectures/timm/families/{family}/variants/{variant}/pretrained-tags")
        assert response.status_code == status.HTTP_200_OK

        tags = response.json()
        assert isinstance(tags, list)
        assert len(tags) > 0
        assert tags == sorted(set(tags))

    def test_get_timm_pretrained_tags_unknown_family_or_version_returns_empty(self, fxt_client: TestClient):
        """Unknown family or version yields an empty list rather than an error."""
        response = fxt_client.get(
            "/api/model_architectures/timm/families/nonexistent-family/variants/v1/pretrained-tags"
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_classification_list_includes_synthetic_timm_card(self, fxt_client: TestClient):
        """The classification model architectures list includes the synthetic timm card entry."""
        response = fxt_client.get("/api/model_architectures?task=classification")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        timm_card = next(arch for arch in data["model_architectures"] if arch["id"] == "image-classification-timm")

        assert timm_card["task"] == "classification"
        assert timm_card["name"] == "Other models (TIMM)"
        assert timm_card["timm_metadata"] is None
        assert timm_card["license"]["name"] == "varies by model"
        assert f"Geti offers {TimmCatalog.count_backbones()} of these models" in timm_card["description"]
        assert timm_card["capabilities"] is not None
        assert timm_card["capabilities"]["xai"] is False
        assert timm_card["capabilities"]["tiling"] is False
        assert timm_card["stats"] is None
        assert timm_card["support_status"] == "active"

    def test_get_timm_manifest_success(self, fxt_client: TestClient, monkeypatch):
        """A valid family/variant/pretrained_tag combination returns a full manifest view."""
        families = fxt_client.get("/api/model_architectures/timm/families").json()
        family = families[0]
        variants = fxt_client.get(f"/api/model_architectures/timm/families/{family}/variants").json()
        variant = variants[0]
        tags = fxt_client.get(
            f"/api/model_architectures/timm/families/{family}/variants/{variant}/pretrained-tags"
        ).json()
        pretrained_tag = tags[0]

        response = fxt_client.get(
            "/api/model_architectures/timm/manifest",
            params={"family": family, "variant": variant, "pretrained_tag": pretrained_tag},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"].startswith("image-classification-timm-")
        assert data["task"] == "classification"
        assert data["timm_metadata"] is not None
        assert data["timm_metadata"]["family"] == family
        assert data["timm_metadata"]["variant"] == variant
        assert data["timm_metadata"]["pretrained_tag"] == pretrained_tag
        assert data["capabilities"]["xai"] is False
        assert data["capabilities"]["tiling"] is False
        assert data["support_status"] == "active"
        assert data["stats"] is not None

    def test_get_timm_manifest_not_found(self, fxt_client: TestClient):
        """An unknown family/variant/pretrained_tag combination returns 404."""
        response = fxt_client.get(
            "/api/model_architectures/timm/manifest",
            params={"family": "nonexistent-family", "variant": "v1", "pretrained_tag": "tag1"},
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "nonexistent-family" in response.json()["detail"]

    @pytest.mark.parametrize("missing_param", ["family", "variant", "pretrained_tag"])
    def test_get_timm_manifest_missing_required_query_param(self, fxt_client: TestClient, missing_param):
        """Omitting any required query param returns 422."""
        params = {"family": "resnet", "variant": "50", "pretrained_tag": "a1"}
        del params[missing_param]

        response = fxt_client.get("/api/model_architectures/timm/manifest", params=params)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestGetModelArchitectureById:
    """Test cases for resolving a single architecture ID, static or timm-derived."""

    def test_resolves_static_architecture_id(self, fxt_client: TestClient):
        """A static manifest ID resolves to its full architecture view, license included."""
        response = fxt_client.get("/api/model_architectures/object-detection-atss-mobilenet-v2")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["id"] == "object-detection-atss-mobilenet-v2"
        assert data["name"] == "ATSS-MobileNet-V2"
        assert data["task"] == "detection"
        assert data["license"]["name"]

    def test_resolves_concrete_timm_backbone_id(self, fxt_client: TestClient):
        """A concrete timm backbone ID (as stored on trained models) resolves with its per-model license."""
        families = fxt_client.get("/api/model_architectures/timm/families").json()
        family = families[0]
        variant = fxt_client.get(f"/api/model_architectures/timm/families/{family}/variants").json()[0]
        pretrained_tag = (
            fxt_client.get(f"/api/model_architectures/timm/families/{family}/variants/{variant}/pretrained-tags").json()
        )[0]
        model_name = TimmCatalog.get_model_name(family, variant, pretrained_tag)
        assert model_name is not None
        architecture_id = model_name_to_id(model_name)

        response = fxt_client.get(f"/api/model_architectures/{architecture_id}")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        expected = TimmManifestProvider.build_manifest(model_name)
        assert data["id"] == architecture_id
        assert data["task"] == "classification"
        assert data["timm_metadata"] is not None
        assert data["license"]["name"] == expected.license.name
        assert data["license"]["url"] == expected.license.url

    def test_unknown_id_returns_404(self, fxt_client: TestClient):
        """An unknown architecture ID returns 404 with a descriptive detail."""
        response = fxt_client.get("/api/model_architectures/not-a-real-architecture")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not-a-real-architecture" in response.json()["detail"]

    def test_unknown_timm_backbone_id_returns_404(self, fxt_client: TestClient):
        """A well-formed but nonexistent timm backbone ID returns 404, not a 500."""
        response = fxt_client.get("/api/model_architectures/image-classification-timm-not-a-real-backbone")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "image-classification-timm-not-a-real-backbone" in response.json()["detail"]

    def test_synthetic_timm_card_id_is_not_resolvable(self, fxt_client: TestClient):
        """The synthetic selector card ID stays unresolvable by design (it selects no concrete backbone)."""
        response = fxt_client.get("/api/model_architectures/image-classification-timm")
        assert response.status_code == status.HTTP_404_NOT_FOUND
