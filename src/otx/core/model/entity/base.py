# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
from torch import nn

from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)
from otx.core.utils.build import get_default_async_reqs_num

if TYPE_CHECKING:
    from omegaconf import DictConfig


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
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        raise NotImplementedError

    def forward(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
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


class OVModel(OTXModel):
    def __init__(self, config: DictConfig) -> None:
        self.model_name = config.pop("model_name")
        self.model_type = config.pop("model_type")
        self.async_inference = config.pop("async_inference", False)
        self.num_requests = config.pop("max_num_requests", get_default_async_reqs_num())
        self.use_tp_mode = config.pop("use_tp_mode", False)
        self.config = config
        super().__init__()

    def _create_model(self) -> nn.Module:
        """Create a OV model with help of Model API."""
        from openvino.model_api.adapters import OpenvinoAdapter, create_core, get_user_config
        from openvino.model_api.models import Model

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_tp_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_adapter = OpenvinoAdapter(
            create_core(),
            self.model_name,
            max_num_requests=self.num_requests,
            plugin_config=plugin_config,
        )

        return Model.create_model(model_adapter, model_type=self.model_type)

    def _customize_inputs(self, entity: T_OTXBatchDataEntity) -> dict[str, Any]:
        # restore original numpy image
        images = [np.transpose(im.numpy(), (1, 2, 0)) for im in entity.images]
        return {"inputs": images}

    def forward(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""

        def _callback(result, user_data) -> None:
            user_data.append(result)

        numpy_inputs = self._customize_inputs(inputs)["inputs"]
        if self.async_inference:
            outputs = []
            self.model.set_callback(_callback)
            for im in numpy_inputs:
                if not self.model.is_ready():
                    self.model.await_any()
                self.model.infer_async(im, user_data=outputs)
            self.model.await_all()
        else:
            outputs = [self.model(im) for im in numpy_inputs]

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != OTXModel._customize_outputs
            else outputs
        )
