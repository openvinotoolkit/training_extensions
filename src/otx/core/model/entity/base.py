# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Union

from torch import nn

from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)


class OTXModel(nn.Module, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """Base class for the models used in OTX."""

    def __init__(self) -> None:
        super().__init__()
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""

    def _customize_inputs(self, inputs: T_OTXBatchDataEntity) -> dict[str, Any]:
        """Customize OTX input batch data entity if needed for you model."""
        raise NotImplementedError

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: T_OTXBatchDataEntity,
    ) -> Union[T_OTXBatchPredEntity, OTXBatchLossEntity]:
        """Customize OTX output batch data entity if needed for you model."""
        raise NotImplementedError

    def forward(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> Union[T_OTXBatchPredEntity, OTXBatchLossEntity]:
        """Model forward function."""
        # If customize_inputs is overrided
        outputs = (
            self.model(**self._customize_inputs(inputs))
            if self._customize_inputs != OTXModel._customize_inputs
            else self.model(inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != OTXModel._customize_outputs
            else outputs
        )
