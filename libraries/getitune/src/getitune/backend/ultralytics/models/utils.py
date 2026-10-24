# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers shared by Ultralytics model wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ultralytics.nn.modules.head import Classify

if TYPE_CHECKING:
    from getitune.backend.ultralytics.models.base import UltralyticsModel


class _MultiLabelClassify(Classify):
    """``Classify`` head variant that emits sigmoid scores during inference.

    Training and export modes continue to emit raw logits so BCE loss and
    export-time post-processing remain consistent.

    Implemented as a genuine subclass (rather than monkey-patching
    ``instance.forward``) so that ``deepcopy``/``torch.save`` — which pickle
    bound methods via ``(getattr, (obj, func.__name__))`` — can correctly
    resolve ``forward`` as a normal class attribute instead of failing to
    find it on the instance.
    """

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Emit sigmoid scores for inference; raw logits during training/export."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training or self.export:
            return x
        return x.sigmoid()


def patch_multilabel_classify_head(model: torch.nn.Module) -> None:
    """Replace each ``Classify`` head's inference softmax with sigmoid.

    This patch is required for multi-label classification because upstream
    YOLO classification checkpoints use softmax semantics by default.

    Reassigns ``__class__`` to a ``Classify`` subclass instead of patching
    ``forward`` on the instance, keeping the module pickle/deepcopy-safe.
    """
    for module in model.modules():
        if isinstance(module, Classify) and not isinstance(module, _MultiLabelClassify):
            module.__class__ = _MultiLabelClassify


def ensure_classify_transforms(model: UltralyticsModel) -> None:
    """Backfill ``model.transforms`` on the underlying ``.pt`` classify checkpoint.

    Ultralytics' ``ClassificationPredictor.setup_source`` unconditionally
    reads ``model.transforms`` for ``.pt`` classification checkpoints (see
    ``ultralytics.models.yolo.classify.predict.ClassificationPredictor``).
    That attribute is normally populated by Ultralytics' own training loop
    (``ClassificationTrainer._setup_train``), but getitune's custom
    trainers bridge data through a ``DataModule`` and always feed
    already-preprocessed tensors, so the attribute may never get set.

    This backfills a stub so Ultralytics does not crash resolving it. The
    actual preprocessing is unaffected: ``ClassificationPredictor.preprocess``
    skips ``self.transforms`` entirely whenever tensor input is provided, so
    this stub is never applied to our data — it only satisfies the
    attribute-existence check.
    """
    yolo_model = model.yolo.model
    if yolo_model is None or hasattr(yolo_model, "transforms"):
        return

    from ultralytics.data.augment import classify_transforms

    yolo_model.transforms = classify_transforms(size=model.imgsz)
