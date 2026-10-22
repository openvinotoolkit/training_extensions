# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Which annotation types the GPU augmentation pipeline transforms, per task.

Pulled out on its own because it's a fact about the task, not about which
backend is running it: `GPUAugmentationCallback` (Lightning) and
`GetiTuneHFTrainer` (Hugging Face) both need exactly the same answer to
"what should Kornia's ``AugmentationSequential`` also transform besides the
image, for this task?"

Instance segmentation uses the ``keypoints`` slot rather than ``mask``: its
annotations are polygons, and transforming ``(N, P, 2)`` coordinates is both
cheaper and exactly what Kornia's keypoint path already does. The packing of
ragged per-sample polygons into the ``(B, M, 2)`` tensor Kornia expects stays
in ``GPUAugmentationPipeline.forward``, since it's a Kornia-shape detail, not a
task fact. Semantic segmentation keeps ``mask`` — it has no polygon carrier.

This module is the single source of truth for that per-task table; both
``GPUAugmentationCallback`` and ``GetiTuneHFTrainer`` import ``DATA_KEYS_BY_TASK``
from here rather than keeping their own copies.
"""

from __future__ import annotations

from getitune.types.task import TaskType

DATA_KEYS_BY_TASK: dict[TaskType, tuple[str, ...]] = {
    TaskType.MULTI_CLASS_CLS: ("label",),
    TaskType.MULTI_LABEL_CLS: ("label",),
    TaskType.DETECTION: ("bbox_xyxy", "label"),
    TaskType.INSTANCE_SEGMENTATION: ("bbox_xyxy", "keypoints", "label"),
    TaskType.KEYPOINT_DETECTION: ("keypoints", "label"),
    TaskType.SEMANTIC_SEGMENTATION: ("mask",),
}
