# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .task import TaskType
from .training_configuration import AlgoLevelParameters


class ModelManifestDeprecationStatus(StrEnum):
    """Status of a model architecture with respect to the deprecation process."""

    ACTIVE = "active"  # Model architecture is fully supported, models can be trained
    DEPRECATED = "deprecated"  # Model architecture is deprecated, can still view and train but it's discouraged
    OBSOLETE = "obsolete"  # Model architecture is no longer supported, models can be still viewed but not trained


class BenchmarkMetrics(BaseModel):
    """Benchmark metrics for different model tasks."""

    model_config = ConfigDict(extra="forbid")

    # Classification metrics
    imagenet_top1_accuracy: float | None = Field(
        default=None,
        title="ImageNet Top-1 Accuracy",
        description="Top-1 accuracy on ImageNet (percentage)",
        ge=0,
        le=100,
    )
    imagenet_top5_accuracy: float | None = Field(
        default=None,
        title="ImageNet Top-5 Accuracy",
        description="Top-5 accuracy on ImageNet (percentage, null if N/A)",
        ge=0,
        le=100,
    )

    # Detection/Segmentation metrics
    coco_map_50_95: float | None = Field(
        default=None,
        title="COCO mAP 50-95",
        description="COCO mean Average Precision at IoU=0.50:0.95 (percentage)",
        ge=0,
        le=100,
    )
    coco_map_50: float | None = Field(
        default=None,
        title="COCO mAP 50",
        description="COCO mean Average Precision at IoU=0.50 (percentage, null if N/A)",
        ge=0,
        le=100,
    )


class ModelStats(BaseModel):
    """Information about a machine learning model."""

    model_config = ConfigDict(extra="forbid")
    gigaflops: float = Field(
        ge=0, title="Gigaflops", description="Billions of floating-point operations per second required by the model"
    )
    trainable_parameters: float = Field(
        ge=0.0,
        default=0.0,
        title="Trainable parameters (millions)",
        description="Number of trainable parameters in the model, expressed in millions",
    )
    benchmark_metrics: BenchmarkMetrics = Field(
        title="Benchmark metrics", description="Standardized benchmark metrics for model performance"
    )


class Capabilities(BaseModel):
    """Model capabilities configuration."""

    model_config = ConfigDict(extra="forbid")
    xai: bool = Field(
        default=False, title="Explainable AI Support", description="Whether the model supports explainable AI features"
    )
    tiling: bool = Field(
        default=False,
        title="Tiling Support",
        description="Whether the model supports image tiling for processing large images",
    )


class WeightsSource(StrEnum):
    """Source of pretrained weights for a model architecture."""

    TIMM = "timm"  # Weights are managed by the timm library
    DIRECT_LINK = "direct_link"  # Weights URL and checksum are defined directly in the manifest file


class DirectLinkPretrainedWeights(BaseModel):
    """Pretrained weights provided via a direct download link."""

    model_config = ConfigDict(extra="forbid")
    source: Literal[WeightsSource.DIRECT_LINK] = WeightsSource.DIRECT_LINK
    url: str = Field(title="Weights URL", description="URL to download the pretrained weights")
    mirror_url: str | None = Field(
        default=None, title="Weights mirror URL", description="Alternative URL to download the weights"
    )
    sha_sum: str = Field(title="Weights SHA256", description="SHA256 checksum of the pretrained weights file")
    cache_filename: str | None = Field(
        default=None,
        title="Cached weights filename",
        description="Filename used when caching the pretrained weights file. "
        "If not provided, the filename will be derived from the URL. Set this when 'url' and 'mirror_url' "
        "resolve to different filenames, so both download sources map to the same local cache entry.",
    )

    @property
    def local_filename(self) -> str:
        """Filename to use when caching the weights locally, regardless of which URL was used to download them."""
        return self.cache_filename or Path(self.url).name


class TimmPretrainedWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal[WeightsSource.TIMM] = WeightsSource.TIMM


PretrainedWeights = Annotated[
    DirectLinkPretrainedWeights | TimmPretrainedWeights,
    Field(discriminator="source"),
]


class TimmMetadata(BaseModel):
    """Metadata specific to timm backbones."""

    model_config = ConfigDict(extra="forbid")
    family: str = Field(title="Model family", description="Family or category of the model architecture")
    variant: str = Field(title="Model variant", description="Specific variant of the model architecture")
    pretrained_tag: str = Field(
        title="Pretrained tag",
        description="Pretrained tag of the model (e.g. timm's pretraining tag such as 'in22k_ft_in22k')",
    )


class ModelManifest(BaseModel):
    """ModelManifest contains the necessary information for training a specific machine learning model."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(title="Model architecture ID", description="Unique identifier for the model architecture")
    name: str = Field(title="Model architecture name", description="Friendly name of the model architecture")
    license: str = Field(
        default="Apache 2.0", title="License", description="License under which the model architecture is released"
    )
    pretrained_weights: PretrainedWeights = Field(
        title="Pretrained Weights", description="Information about the pretrained weights for the model architecture"
    )
    timm_metadata: TimmMetadata | None = Field(
        default=None, title="TIMM Metadata", description="Metadata specific to timm backbones"
    )
    description: str = Field(title="Description", description="Detailed description of the model capabilities")
    task: TaskType = Field(title="Task Type", description="Type of machine learning task addressed by the model")
    stats: ModelStats = Field(title="Model Statistics", description="Statistics about the model")
    support_status: ModelManifestDeprecationStatus = Field(
        default=ModelManifestDeprecationStatus.ACTIVE,
        title="Support Status",
        description="Current support level (active, deprecated, or obsolete)",
    )
    capabilities: Capabilities = Field(
        title="Model Capabilities", description="Special capabilities supported by the model"
    )
    hyperparameters: AlgoLevelParameters = Field(
        title="Hyperparameters", description="Configuration parameters for model training"
    )

    @model_validator(mode="after")
    def validate_metrics(self) -> "ModelManifest":
        if self.task in {TaskType.CLASSIFICATION}:
            if (
                isinstance(self.pretrained_weights, DirectLinkPretrainedWeights)
                and self.stats.benchmark_metrics.imagenet_top1_accuracy is None
            ):
                raise ValueError("For classification 'imagenet_top1_accuracy' benchmark metric is required")
            if (
                self.stats.benchmark_metrics.coco_map_50 is not None
                or self.stats.benchmark_metrics.coco_map_50_95 is not None
            ):
                raise ValueError(
                    "For classification, only ImageNet Accuracy benchmark metrics can be set. However, one of the COCO "
                    "mAP benchmark metrics was not 'None'."
                )
        if self.task in {TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION}:
            if self.stats.benchmark_metrics.coco_map_50_95 is None:
                raise ValueError("For detection or instance segmentation 'coco_map_50_95' benchmark metric is required")
            if (
                self.stats.benchmark_metrics.imagenet_top1_accuracy is not None
                or self.stats.benchmark_metrics.imagenet_top5_accuracy is not None
            ):
                raise ValueError(
                    "For detection or instance segmentation, only COCO mAP benchmark metrics can be set. However, one "
                    "of the ImageNet Accuracy benchmark metrics was not 'None'."
                )
        return self
