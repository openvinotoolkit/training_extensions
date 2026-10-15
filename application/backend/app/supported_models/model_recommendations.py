# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from app.models import TaskType


class ModelArchitectureCategory(StrEnum):
    BALANCE = "balance"
    SPEED = "speed"
    ACCURACY = "accuracy"


RECOMMENDED_MODEL_ARCHITECTURES = {
    TaskType.CLASSIFICATION: {
        ModelArchitectureCategory.BALANCE: "image-classification-vit-tiny",
        ModelArchitectureCategory.ACCURACY: "image-classification-dinov2",
        ModelArchitectureCategory.SPEED: "image-classification-mobilenet-v3-large",
    },
    TaskType.DETECTION: {
        ModelArchitectureCategory.BALANCE: "object-detection-dfine-m",
        ModelArchitectureCategory.ACCURACY: "object-detection-dfine-l",
        ModelArchitectureCategory.SPEED: "object-detection-yolox-s",
    },
    TaskType.INSTANCE_SEGMENTATION: {
        ModelArchitectureCategory.BALANCE: "instance-segmentation-rfdetr-m",
        ModelArchitectureCategory.ACCURACY: "instance-segmentation-rfdetr-xl",
        ModelArchitectureCategory.SPEED: "instance-segmentation-rfdetr-s",
    },
}
