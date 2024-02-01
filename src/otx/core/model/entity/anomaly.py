"""Base Anomaly OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.core.model.entity.base import OTXModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch import nn


class _AnomalibLightningArgsCache:
    """Caches args for the anomalib lightning module.

    This is needed as the arguments are passed to the OTX model. These are saved and used by the OTX anomaly
    lightning model.
    """

    def __init__(self):
        self._args: dict[str, Any] = {}

    def update(self, **kwargs) -> None:
        """Add args to cache."""
        self._args.update(kwargs)

    def get(self) -> dict[str, Any]:
        """Get cached args."""
        return self._args


class OTXAnomalyModel(OTXModel):
    """Base Anomaly OTX Model."""

    def __init__(
        self,
    ) -> None:
        self.model: nn.Module
        super().__init__(num_classes=2)
        # This cache is used to get params from the OTX model and pass it into Anomalib Lightning module
        self.anomalib_lightning_args = _AnomalibLightningArgsCache()

    def _customize_inputs(self, *_, **__) -> None:
        """Input customization is done through the lightning module."""

    def _customize_outputs(self, *_, **__) -> None:
        """Output customization is done through the lightning module."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Call forward on the raw tensor.

        Overrides the base forward as input and output customization occurs from the lightning model.
        """
        return self.model(input_tensor)

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        # TODO
        ...
