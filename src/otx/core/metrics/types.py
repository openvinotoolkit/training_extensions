# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Type definitions for OTX metrics."""

import logging
from typing import Callable

from torch import Tensor, zeros
from torchmetrics import Metric, MetricCollection

from otx.core.types.label import LabelInfo

MetricCallable = Callable[[LabelInfo], Metric | MetricCollection]


class NullMetric(Metric):
    """Null metric."""

    def update(self, *args, **kwargs) -> None:
        """Do not update."""
        return

    def compute(self) -> dict:
        """Return a null metric result."""
        msg = "NullMetric does not report any valid metric. Please change this to appropriate metric if needed."
        logging.warning(msg)
        return {"null_metric": zeros(size=[0], device=self.device)}


def _null_metric_callable(_: LabelInfo) -> Metric:
    return NullMetric()


NullMetricCallable = _null_metric_callable

# TODO(vinnamki): Remove the second typing list[dict[str, Tensor]] coming from semantic seg task if possible
MetricInput = dict[str, list[dict[str, Tensor]]] | list[dict[str, Tensor]]
