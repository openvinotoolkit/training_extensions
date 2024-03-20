"""OTX Draem model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from anomalib.models.image import Draem as AnomalibDraem

from otx.core.model.entity.base import OTXModel
from otx.core.model.module.anomaly import OTXAnomaly


class Draem(OTXAnomaly, OTXModel, AnomalibDraem):
    """OTX Draem model.

    Args:
        enable_sspcab (bool): Enable SSPCAB training. Defaults to ``False``.
        sspcab_lambda (float): SSPCAB loss weight. Defaults to ``0.1``.
        anomaly_source_path (str | None): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty. Defaults to ``None``.
        beta (float | tuple[float, float]): Parameter that determines the opacity of the noise mask.
            Defaults to ``(0.1, 1.0)``.
    """

    def __init__(
        self,
        enable_sspcab: bool = False,
        sspcab_lambda: float = 0.1,
        anomaly_source_path: str | None = None,
        beta: float | tuple[float, float] = (0.1, 1.0),
        num_classes: int = 2,
    ) -> None:
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, num_classes=num_classes)
        AnomalibDraem.__init__(
            self,
            enable_sspcab=enable_sspcab,
            sspcab_lambda=sspcab_lambda,
            anomaly_source_path=anomaly_source_path,
            beta=beta,
        )
