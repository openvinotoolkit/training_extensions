# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DinoV2Seg model implementations."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import urlparse

from torch.hub import download_url_to_file

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.vision_transformer import VisionTransformerBackbone
from getitune.backend.lightning.models.segmentation.base import LightningSegmentationModel
from getitune.backend.lightning.models.segmentation.heads import FCNHead
from getitune.backend.lightning.models.segmentation.losses import CrossEntropyLossWithIgnore
from getitune.backend.lightning.models.segmentation.segmentors import BaseSegmentationModel
from getitune.backend.lightning.models.utils.utils import load_checkpoint
from getitune.config.data import TileConfig
from getitune.metrics.dice import SegmCallable
from getitune.types import PathLike

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types.label import LabelInfoTypes


class DinoV2Seg(LightningSegmentationModel):
    """DinoV2Seg for Semantic Segmentation model.

    Args:
        label_info (LabelInfoTypes): Information about the hierarchical labels.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
        model_name (Literal, optional): Name of the model. Defaults to "dinov2-small-seg".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
        Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to SegmCallable.
        torch_compile (bool, optional): Flag to indicate whether to use torch.compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    pretrained_urls: ClassVar[dict[str, str]] = {
        "dinov2-small-seg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth,"
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_linear_head.pth"
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["dinov2-small-seg"] = "dinov2-small-seg",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
    ):
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
        )

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        # initialize backbones
        num_classes = num_classes if num_classes is not None else self.num_classes

        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        backbone = VisionTransformerBackbone(model_name=self.model_name, img_size=self.data_input_params.input_size)
        backbone.forward = partial(  # type: ignore[method-assign]
            backbone.get_intermediate_layers,
            n=[8, 9, 10, 11],
            reshape=True,
        )
        decode_head = FCNHead(self.model_name, num_classes=num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]

        backbone.init_weights()

        return BaseSegmentationModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}

    def load_pretrained(self, weights: PathLike | None = None) -> None:
        """Load pretrained weights for the model.

        The DinoV2Seg backbone and decode head are pretrained separately, so ``weights`` is expected
        to encode both locations as a single comma-separated string: ``"<backbone_weights>,<head_weights>"``.
        Each part may be a local file path or a URL, and is downloaded to
        ``$PRETRAINED_WEIGHTS_CACHE_DIR`` on first use if not already present locally.

        Args:
            weights (PathLike | None, optional): Comma-separated ``"<backbone_weights>,<head_weights>"``
                string pointing to the backbone and decode head weights (local paths or URLs).
                If None, falls back to ``self.pretrained_urls[self.model_name]``. Defaults to None.

        Raises:
            ValueError: If ``weights`` (or the resolved default from ``pretrained_urls``) is not a
                comma-separated ``"<backbone_weights>,<head_weights>"`` string.
        """
        if weights is None:
            weights = self.pretrained_urls[self.model_name]

        if "," not in str(weights):
            msg = "weights must be a comma-separated string of the form '<backbone_weights>,<head_weights>'"
            raise ValueError(msg)

        backbone_weights, head_weights = str(weights).split(",")

        def _download_from_url(weights: str) -> Path:
            import os

            parts = urlparse(weights)
            filename = Path(parts.path).name
            weights_path = Path(os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"]) / filename
            if not weights_path.exists():
                download_url_to_file(str(weights), str(weights_path), "", progress=True)
            return weights_path

        if not Path(backbone_weights).exists():
            backbone_weights = _download_from_url(backbone_weights)
        if not Path(head_weights).exists():
            head_weights = _download_from_url(head_weights)

        backbone = cast("VisionTransformerBackbone", self.model.backbone)
        backbone.load_checkpoint(backbone_weights)  # pyrefly: ignore[not-callable]
        load_checkpoint(cast("nn.Module", self.model.decode_head), str(head_weights), prefix="decode_head")

        # freeze backbone
        for _, v in backbone.named_parameters():
            v.requires_grad = False
