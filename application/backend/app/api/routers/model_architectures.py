# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing model architectures"""

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.model_architecture import ModelArchitectures, ModelArchitectureView, TopPicks
from app.models import TaskType
from app.models.model_manifest import Capabilities, ModelManifestDeprecationStatus
from app.services import ModelManifestService
from app.supported_models import RECOMMENDED_MODEL_ARCHITECTURES
from app.supported_models.timm import TimmCatalog, TimmManifestProvider

router = APIRouter(prefix="/api/model_architectures", tags=["Model Architectures"])


@router.get(
    "",
    response_model=ModelArchitectures,
    responses={
        status.HTTP_200_OK: {"description": "List of available model architectures"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid task type provided"},
    },
)
def get_model_architectures(task: TaskType) -> ModelArchitectures:
    """
    Get all available model architectures, optionally filtered by task type.

    Args:
        task: Task type filter (e.g., 'detection', 'classification', 'instance_segmentation')

    Returns:
        ModelArchitectures containing list of model architectures and recommended top picks.
    """

    model_manifests = ModelManifestService.get_model_manifests()
    top_picks = RECOMMENDED_MODEL_ARCHITECTURES.get(task, None)

    architectures = [
        ModelArchitectureView.model_validate(manifest, from_attributes=True)
        for manifest in model_manifests.values()
        if manifest.task == task
    ]
    if task == TaskType.CLASSIFICATION:
        architectures.append(_build_timm_card_entry())

    return ModelArchitectures(
        model_architectures=architectures,
        top_picks=TopPicks.model_validate(top_picks, from_attributes=True),
    )


@router.get("/timm/families", response_model=list[str])
def get_timm_families() -> list[str]:
    """Get all available timm architecture families."""
    return TimmCatalog.list_families()


@router.get("/timm/families/{family}/variants", response_model=list[str])
def get_timm_variants(family: str) -> list[str]:
    """Get all available timm architecture variants for a given family."""
    return TimmCatalog.list_variants(family)


@router.get("/timm/families/{family}/variants/{variant}/pretrained-tags", response_model=list[str])
def get_timm_pretrained_tags(family: str, variant: str) -> list[str]:
    """Get all available timm pretrained tags for a given family and version."""
    return TimmCatalog.list_pretrained_tags(family, variant)


@router.get(
    "/timm/manifest",
    response_model=ModelArchitectureView,
    responses={
        status.HTTP_200_OK: {"description": "TIMM model architecture manifest"},
        status.HTTP_404_NOT_FOUND: {"description": "No timm model found"},
    },
)
def get_timm_manifest(family: str, variant: str, pretrained_tag: str) -> ModelArchitectureView:
    """Build and return the TIMM manifest view for a specific family, variant, and pretrained tag."""
    model_name = TimmCatalog.get_model_name(family, variant, pretrained_tag)
    if model_name is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No timm model found for family '{family}', variant '{variant}', and pretrained "
            f"tag '{pretrained_tag}'.",
        )
    manifest = TimmManifestProvider.build_manifest(model_name)
    return ModelArchitectureView.model_validate(manifest, from_attributes=True)


TIMM_CARD_ID = "image-classification-timm"


def _build_timm_card_entry() -> ModelArchitectureView:
    """Build the synthetic 'Custom backbone (timm)' card entry.

    This is a fixed, non-trainable entry point rendered by the UI to open the
    searchable timm backbone selector. It is intentionally never registered in
    `ModelManifestService.get_model_manifests()` and its id is never resolvable
    via `get_model_manifest_by_id` — selecting a concrete backbone always uses
    a real `image-classification-timm-<name>` id instead.
    """
    description = (
        "PyTorch Image Models (TIMM) is a large collection of SOTA image classification models and "
        f"pretrained weights. Geti offers {TimmCatalog.count_backbones()} of these models across dozens "
        "of architecture families."
    )
    return ModelArchitectureView(
        id=TIMM_CARD_ID,
        task=TaskType.CLASSIFICATION,
        name="Other models (TIMM)",
        timm_metadata=None,
        description=description,
        capabilities=Capabilities(xai=False, tiling=False),
        license="varies by model",
        stats=None,
        support_status=ModelManifestDeprecationStatus.ACTIVE,
    )
