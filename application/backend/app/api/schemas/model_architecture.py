# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from app.models import TaskType
from app.models.model_manifest import Capabilities, ModelManifestDeprecationStatus, ModelStats, TimmMetadata


class ModelArchitectureView(BaseModel):
    """Simplified model architecture information for API responses"""

    id: str = Field(title="Model architecture ID", description="Unique identifier for the model architecture")
    task: TaskType = Field(title="Task Type", description="Type of machine learning task addressed by the model")
    name: str = Field(title="Model architecture name", description="Friendly name of the model architecture")
    description: str = Field(title="Description", description="Detailed description of the model capabilities")
    timm_metadata: TimmMetadata | None = Field(
        default=None, title="TIMM Model Metadata", description="Metadata about the specific TIMM model architecture"
    )
    capabilities: Capabilities = Field(
        title="Model Capabilities", description="Special capabilities supported by the model"
    )
    license: str = Field(title="License", description="License under which the model architecture is released")
    stats: ModelStats | None = Field(title="Model Statistics", description="Statistics about the model")
    support_status: ModelManifestDeprecationStatus = Field(
        title="Support Status", description="Current support level (active, deprecated, or obsolete)"
    )


class TopPicks(BaseModel):
    """Top picks for model architectures based on categories"""

    balance: str = Field(title="Balance Model", description="Model architecture that balances accuracy and speed")
    speed: str = Field(title="Speed Model", description="Model architecture optimized for speed")
    accuracy: str = Field(title="Accuracy Model", description="Model architecture optimized for accuracy")


class ModelArchitectures(BaseModel):
    """Model architectures response"""

    model_architectures: list[ModelArchitectureView] = Field(
        title="Model Architectures", description="List of available model architectures"
    )

    top_picks: TopPicks | None = Field(
        title="Top Picks", description="Recommended model architectures for different categories"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_architectures": [
                    {
                        "id": "object-detection-dfine-m",
                        "task": "detection",
                        "name": "D-FINE-M",
                        "license": "Apache 2.0",
                        "description": "D-FINE is a powerful real-time object detector that redefines the bounding box"
                        " regression task in DETRs as Fine-grained Distribution Refinement (FDR)."
                        " Combined with the DEIM adaptive augmentation scheduling framework"
                        " (enabled by default), it achieves outstanding performance with faster convergence.",
                        "capabilities": {"tiling": False},
                        "stats": {
                            "gigaflops": 57,
                            "trainable_parameters": 19,
                            "performance_ratings": {"accuracy": 2, "training_time": 2, "inference_speed": 2},
                        },
                        "support_status": "active",
                    },
                ],
                "top_picks": {
                    "balance": "object-detection-atss-mobilenet-v2",
                    "speed": "object-detection-yolox-s",
                    "accuracy": "object-detection-dfine-l",
                },
            }
        }
    }
