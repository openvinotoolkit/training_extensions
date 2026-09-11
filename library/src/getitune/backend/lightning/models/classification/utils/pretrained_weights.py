# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pretrained-weight loader mixins for classification models.

Each mixin implements ``load_pretrained`` for one download backend and operates on ``self.model.backbone``.
Mix into a model *before* the task base class so the mixin's ``load_pretrained`` overrides the base no-op.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlparse

from torch.hub import download_url_to_file

from getitune.backend.lightning.models.utils.utils import (
    load_checkpoint,
)

if TYPE_CHECKING:
    from torch import nn

    from getitune.backend.lightning.models.classification.backbones.vision_transformer import (
        VisionTransformerBackbone,
    )
    from getitune.types import PathLike

logger = logging.getLogger(__name__)


class _ClassifierModel(Protocol):
    """A classifier exposing a ``backbone`` submodule."""

    backbone: nn.Module


class _SupportsBackboneWeights(Protocol):
    """Lightning classification model wrapper exposing a classifier backbone."""

    model: _ClassifierModel
    model_name: str


class _ViTClassifierModel(Protocol):
    """A classifier exposing a ViT backbone."""

    backbone: VisionTransformerBackbone


class _SupportsViTBackboneWeights(Protocol):
    """Lightning classification model wrapper exposing a ViT backbone."""

    pretrained_urls: dict[str, str]
    model: _ViTClassifierModel
    model_name: str


class PytorchcvWeightsLoader:
    """Load backbone weights via pytorchcv's model store (EfficientNet)."""

    def load_pretrained(self: _SupportsBackboneWeights, weights: PathLike | None = None) -> None:
        """Download EfficientNet backbone weights into the cache dir."""
        import zipfile

        from pytorchcv.models.common.model_store import download_model

        weights_path = Path(weights) if weights is not None else None
        if weights_path is not None and weights_path.suffix == ".zip" and weights_path.is_file():
            with zipfile.ZipFile(weights_path) as zf:
                zf.extractall(weights_path.parent)
            weights_path = weights_path.with_suffix("")

        cache_dir = str(weights_path.parent) if weights_path is not None else os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"]
        download_model(
            net=self.model.backbone,
            model_name=self.model_name,
            local_model_store_dir_path=cache_dir,
        )
        logger.info("Loaded backbone weights from %s", cache_dir)


class TorchvisionWeightsLoader:
    """Load backbone weights from Torchvision (EfficientNet)."""

    def load_pretrained(self: _SupportsBackboneWeights, weights: PathLike | None = None) -> None:
        """Load weights: a local checkpoint if given, else torchvision's official set."""
        if weights is not None and Path(weights).exists():
            load_checkpoint(self.model.backbone, str(weights))
            return

        from torchvision.models import get_model, get_model_weights

        ref = get_model(name=self.model_name, weights=get_model_weights(self.model_name).verify("DEFAULT"))
        self.model.backbone.features.load_state_dict(ref.features.state_dict())  # pyrefly: ignore[missing-attribute]


class TimmWeightsLoader:
    """Load backbone weights via ``timm.models.load_pretrained``."""

    def load_pretrained(self: _SupportsBackboneWeights, weights: PathLike | None = None) -> None:
        """Load weights: a local checkpoint if given, else timm's pretrained source."""
        timm_model = cast("nn.Module", self.model.backbone.model)  # the nn.Module created by timm.create_model

        if weights is not None and Path(weights).exists():
            load_checkpoint(timm_model, str(weights))
            return

        from timm.models import load_pretrained

        load_pretrained(
            timm_model,
            pretrained_cfg=timm_model.pretrained_cfg,  # pyrefly: ignore[bad-argument-type]
            num_classes=0,
        )
        logger.info("Loaded timm pretrained weights for %s", self.model_name)


class VisionTransformerWeightsLoader:
    """Load backbone weights for ViT architecture."""

    def load_pretrained(self: _SupportsViTBackboneWeights, weights: PathLike | None = None) -> None:
        """Load weights: a local checkpoint if given, else download from ``self.pretrained_urls`` into the cache dir."""
        if weights is None or not Path(weights).exists():
            if self.model_name not in self.pretrained_urls:
                warnings.warn(
                    "No pretrained weights found for the specified model. Initializing model with random weights.",
                    stacklevel=1,
                )
                return

            pretrained_url = self.pretrained_urls[self.model_name]
            logger.info("init weight - %s", pretrained_url)
            parts = urlparse(pretrained_url)
            filename = Path(parts.path).name

            cache_dir = Path(os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"])
            weights = cache_dir / filename
            if not Path.exists(weights):
                download_url_to_file(pretrained_url, str(weights), "", progress=True)

        self.model.backbone.load_checkpoint(checkpoint_path=Path(weights))  # pyrefly: ignore[not-callable]
        logger.info("Loaded ViT backbone weights from %s", weights)
