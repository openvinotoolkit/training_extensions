# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX Frechet Inception Distance (FID) metric used for diffusion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchmetrics import MetricCollection
from torchmetrics.image.fid import FrechetInceptionDistance

if TYPE_CHECKING:
    from otx.core.types.label import LabelInfo


def _diffusion_metric_callable(
    _: LabelInfo,
) -> MetricCollection:
    return MetricCollection({"fid": FrechetInceptionDistance(normalize=True, reset_real_features=False)})


DiffusionMetricCallable = _diffusion_metric_callable
