# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from app.execution.common.recipe_resolver import RecipeResolver
from app.supported_models.timm import manifest_provider


class TestRecipeResolverTimmRouting:
    @pytest.fixture
    def resolver(self, tmp_path: Path) -> RecipeResolver:
        # Create just enough of the tree for path resolution + existence checks.
        (tmp_path / "classification" / "multi_class_cls").mkdir(parents=True)
        (tmp_path / "classification" / "multi_class_cls" / "timm_generic.yaml").touch()
        (tmp_path / "classification" / "multi_label_cls").mkdir(parents=True)
        (tmp_path / "classification" / "multi_label_cls" / "timm_generic.yaml").touch()
        return RecipeResolver(tmp_path)

    @pytest.mark.parametrize("sub_task_type", ["MULTI_CLASS_CLS", "MULTI_LABEL_CLS"])
    def test_timm_id_routes_to_timm_generic(self, resolver: RecipeResolver, sub_task_type: str) -> None:
        path = resolver.resolve("image-classification-timm-resnet18.a1_in1k", sub_task_type)
        assert path.name == "timm_generic.yaml"
        assert path.parent.name == sub_task_type.lower()

    def test_timm_id_without_sub_task_type_falls_through_to_registry(self, resolver: RecipeResolver) -> None:
        with (
            patch.object(manifest_provider.TimmManifestProvider, "is_timm_id", return_value=True),
            pytest.raises(KeyError),
        ):
            resolver.resolve("image-classification-timm-resnet18.a1_in1k", None)
