# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Tuple

import onnx
import openvino
import torch
from torch import nn

from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)
from otx.core.types.export import OTXExportFormatType, OTXExportPrecisionType

if TYPE_CHECKING:
    import torch


class OTXModel(nn.Module, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """Base class for the models used in OTX."""

    def __init__(self) -> None:
        super().__init__()
        self.classification_layers: list[str] = []
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

    def register_load_state_dict_pre_hook(self, model_classes: list[str], ckpt_classes: list[str]) -> None:
        """Register load_state_dict_pre_hook.

        Args:
            model_classes (list[str]): Class names from training data.
            ckpt_classes (list[str]): Class names from checkpoint state dictionary.
        """
        self.model_classes = model_classes
        self.ckpt_classes = ckpt_classes
        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)

    def load_state_dict_pre_hook(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        """Modify input state_dict according to class name matching before weight loading."""
        model2ckpt = self.map_class_names(self.model_classes, self.ckpt_classes)

        for param_name in self.classification_layers:
            model_param = self.state_dict()[param_name].clone()
            ckpt_param = state_dict[prefix + param_name]
            for model_t, ckpt_t in enumerate(model2ckpt):
                if ckpt_t >= 0:
                    model_param[model_t].copy_(ckpt_param[ckpt_t])

            # Replace checkpoint weight by mixed weights
            state_dict[prefix + param_name] = model_param

    @staticmethod
    def map_class_names(src_classes: list[str], dst_classes: list[str]) -> list[int]:
        """Computes src to dst index mapping.

        src2dst[src_idx] = dst_idx
        #  according to class name matching, -1 for non-matched ones
        assert(len(src2dst) == len(src_classes))
        ex)
          src_classes = ['person', 'car', 'tree']
          dst_classes = ['tree', 'person', 'sky', 'ball']
          -> Returns src2dst = [1, -1, 0]
        """
        src2dst = []
        for src_class in src_classes:
            if src_class in dst_classes:
                src2dst.append(dst_classes.index(src_class))
            else:
                src2dst.append(-1)
        return src2dst

    @abstractmethod
    def _embed_model_metadata(self, model: Any, ) -> Any:
        """Embeds metadata to the exported model"""
        return model

    def export(self, input_size: Tuple[int, int], save_path: str, format: OTXExportFormatType = "OPENVINO",
               precision: OTXExportPrecisionType = "FP32", mean: Tuple[float, float,float] = (0., 0., 0.),
               std: Tuple[float, float,float] = (1., 1., 1.), resize_mode: str = "standard", pad_value: int = 0, swap_rgb: bool = False,
               label_names: List[str] = [], label_ids: List[str] = []) -> None:
        """Export model to a deployable format.

        The resulting model is ready to be executed via ModelAPI."""

        dummy_tensor = torch.rand((1, 3, *input_size)).to(next(self.model.parameters()).device)
        if format == OTXExportFormatType.OPENVINO:
            exported_model = openvino.convert_model(self.model, example_input=dummy_tensor)
            exported_model = self._embed_model_metadata(exported_model)
            openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXExportPrecisionType.FP16))
        elif format == OTXExportFormatType.ONNX:
            torch.onnx.export(self.model, dummy_tensor, save_path)
            onnx_model = onnx.load(save_path)
            onnx_model = self._embed_model_metadata(onnx_model)
            if precision == OTXExportPrecisionType.FP16:
                from onnxconverter_common import float16
                onnx_model = float16.convert_float_to_float16(onnx_model)
            onnx.save(onnx_model, save_path)
