# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.segmentation import MMSegCompatibleModel

if TYPE_CHECKING:
    from pathlib import Path


class DinoV2Seg(MMSegCompatibleModel):
    """DinoV2Seg Model."""

    def __init__(self, num_classes: int) -> None:
        model_name = "dino_v2_seg"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    def export(self, *args) -> Path:
        """Export method for DinoV2Seg.

        Model doesn't support export for now due to unsupported operations from xformers.
        This method will raise an error.
        """
        msg = "DinoV2Seg cannot be exported. It is not supported."
        raise RuntimeError(msg)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}
