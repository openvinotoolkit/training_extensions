# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType
from unittest.mock import patch

import pytest

from app.supported_models.timm import TimmCatalog, catalog

_FAKE_BROWSING_SNAPSHOT = MappingProxyType(
    {
        "resnet18.a1_in1k": {
            "model_name": "resnet18.a1_in1k",
            "family": "resnet",
            "version": "resnet18",
            "pretrained": "a1_in1k",
        },
        "resnet18.a2_in1k": {
            "model_name": "resnet18.a2_in1k",
            "family": "resnet",
            "version": "resnet18",
            "pretrained": "a2_in1k",
        },
        "resnet50.a1_in1k": {
            "model_name": "resnet50.a1_in1k",
            "family": "resnet",
            "version": "resnet50",
            "pretrained": "a1_in1k",
        },
        "vit_base.augreg": {
            "model_name": "vit_base.augreg",
            "family": "vit",
            "version": "vit_base",
            "pretrained": "augreg",
        },
    }
)


class TestTimmCatalog:
    """Tests for TimmCatalog against a small, deterministic fixture snapshot."""

    @pytest.fixture(autouse=True)
    def _fake_browsing_snapshot(self):
        catalog._snapshot.cache_clear()
        with patch.object(catalog, "_snapshot", return_value=_FAKE_BROWSING_SNAPSHOT):
            yield

    def test_list_families_returns_sorted_distinct_names(self) -> None:
        assert TimmCatalog.list_families() == ["resnet", "vit"]

    @pytest.mark.parametrize(
        "family, expected_variants",
        [
            ("resnet", ["resnet18", "resnet50"]),
            ("vit", ["vit_base"]),
            ("unknown", []),
        ],
    )
    def test_list_variants_scoped_to_family(self, family: str, expected_variants: list[str]) -> None:
        assert TimmCatalog.list_variants(family) == expected_variants

    @pytest.mark.parametrize(
        "family, variant, expected_tags",
        [
            ("resnet", "resnet18", ["a1_in1k", "a2_in1k"]),
            ("resnet", "resnet50", ["a1_in1k"]),
            ("resnet", "resnet999", []),
        ],
    )
    def test_list_pretrained_tags_returns_sorted_tags(
        self, family: str, variant: str, expected_tags: list[str]
    ) -> None:
        assert TimmCatalog.list_pretrained_tags(family, variant) == expected_tags

    @pytest.mark.parametrize(
        "family, variant, pretrained_tag, expected_name",
        [
            ("resnet", "resnet18", "a1_in1k", "resnet18.a1_in1k"),
            ("resnet", "resnet18", "a2_in1k", "resnet18.a2_in1k"),
            ("resnet", "resnet50", "a1_in1k", "resnet50.a1_in1k"),
            ("vit", "vit_base", "augreg", "vit_base.augreg"),
            ("resnet", "resnet999", "a1_in1k", None),
            ("unknown", "resnet18", "a1_in1k", None),
        ],
    )
    def test_get_model_name_returns_expected_model_name(
        self, family: str, variant: str, pretrained_tag: str, expected_name: str | None
    ) -> None:
        assert TimmCatalog.get_model_name(family, variant, pretrained_tag) == expected_name
