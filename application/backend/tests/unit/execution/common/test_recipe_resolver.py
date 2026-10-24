# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from app.execution.common.recipe_resolver import RecipeResolver
from app.supported_models.timm import manifest_provider

GETITUNE_RECIPE_ROOT = Path(__file__).parents[6] / "libraries" / "getitune" / "src" / "getitune" / "recipe"

HF_CLASSIFICATION_RECIPES = {
    "image-classification-dinov3-vits": "dinov3_vits.yaml",
    "image-classification-dinov3-vitb16": "dinov3_vitb16.yaml",
    "image-classification-dinov3-vitl16": "dinov3_vitl16.yaml",
    "image-classification-dinov3-convnext-tiny": "dinov3_convnext_tiny.yaml",
    "image-classification-dinov3-convnext-small": "dinov3_convnext_small.yaml",
    "image-classification-dinov3-convnext-base": "dinov3_convnext_base.yaml",
    "image-classification-dinov3-convnext-large": "dinov3_convnext_large.yaml",
    "image-classification-convnextv2-atto": "convnextv2_atto.yaml",
    "image-classification-convnextv2-base": "convnextv2_base.yaml",
    "image-classification-convnextv2-large": "convnextv2_large.yaml",
}

HF_TASK_RECIPES = {
    "object-detection-rtdetrv2-r18": "detection/rtdetrv2_r18.yaml",
    "object-detection-rtdetrv2-r34": "detection/rtdetrv2_r34.yaml",
    "object-detection-rtdetrv2-r50": "detection/rtdetrv2_r50.yaml",
    "object-detection-rtdetrv2-r101": "detection/rtdetrv2_r101.yaml",
    "instance-segmentation-mask2former-swin-s": "instance_segmentation/mask2former_swin_s.yaml",
    "instance-segmentation-mask2former-swin-b": "instance_segmentation/mask2former_swin_b.yaml",
    "instance-segmentation-mask2former-swin-l": "instance_segmentation/mask2former_swin_l.yaml",
    "instance-segmentation-eomt-dinov3-large-640": "instance_segmentation/eomt_dinov3_large_640.yaml",
}


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


@pytest.mark.parametrize("manifest_id, recipe_name", HF_CLASSIFICATION_RECIPES.items())
@pytest.mark.parametrize("sub_task_type", ["MULTI_CLASS_CLS", "MULTI_LABEL_CLS"])
def test_hf_classification_routes_shared_id_to_subtask_recipe(
    manifest_id: str, recipe_name: str, sub_task_type: str
) -> None:
    path = RecipeResolver(GETITUNE_RECIPE_ROOT).resolve(manifest_id, sub_task_type)

    assert path == GETITUNE_RECIPE_ROOT / "classification" / sub_task_type.lower() / recipe_name
    assert yaml.safe_load(path.read_text())["backend"] == "huggingface"


@pytest.mark.parametrize("manifest_id, relative_path", HF_TASK_RECIPES.items())
def test_hf_task_mapping_resolves_to_huggingface_recipe(manifest_id: str, relative_path: str) -> None:
    path = RecipeResolver(GETITUNE_RECIPE_ROOT).resolve(manifest_id, None)

    assert path == GETITUNE_RECIPE_ROOT / relative_path
    assert yaml.safe_load(path.read_text())["backend"] == "huggingface"


@pytest.mark.parametrize(
    "removed_id",
    [
        "image-classification-dinov3-vith16plus",
        "image-classification-dinov3-vit7b16",
        "image-classification-convnextv2-femto",
        "image-classification-convnextv2-huge",
        "instance-segmentation-eomt-large-1280",
        "instance-segmentation-eomt-dinov3-large-1280",
    ],
)
def test_removed_hf_variant_is_not_mapped(removed_id: str) -> None:
    assert removed_id not in RecipeResolver.TEMPLATE_ID_MAPPING
