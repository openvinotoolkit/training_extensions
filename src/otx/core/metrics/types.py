# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Type definitions for OTX metrics."""

from typing import Callable

from torch import Tensor
from torchmetrics import Metric, MetricCollection

from otx.core.types.label import LabelInfo

MetricCallable = Callable[[LabelInfo], Metric | MetricCollection]
NullMetricCallable: MetricCallable = lambda label_info: Metric()  # noqa: ARG005
# TODO(vinnamki): Remove the second typing list[dict[str, Tensor]] coming from semantic seg task if possible
MetricInput = dict[str, list[dict[str, Tensor]]] | list[dict[str, Tensor]]
