# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Tuple

import onnx
import openvino
import torch
from torch import nn

from otx.core.data.dataset.base import LabelInfo
from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)
from otx.core.types.export import OTXExportFormatType, OTXExportPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    import torch


class OTXModel(nn.Module, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """Base class for the models used in OTX.

    Args:
        num_classes: Number of classes this model can predict.
    """

    _EXPORTED_MODEL_BASE_NAME = "exported_model"

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self._label_info = LabelInfo.from_num_classes(num_classes)
        self.classification_layers: dict[str, dict[str, Any]] = {}
        self.model = self._create_model()

    @property
    def label_info(self) -> LabelInfo:
        """Get this model label information."""
        return self._label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfo | list[str]) -> None:
        """Set this model label information."""
        if isinstance(label_info, list):
            label_info = LabelInfo(label_names=label_info)

        old_num_classes = self._label_info.num_classes
        new_num_classes = label_info.num_classes

        if old_num_classes != new_num_classes:
            msg = (
                f"Given LabelInfo has the different number of classes "
                f"({old_num_classes}!={new_num_classes}). "
                "The model prediction layer is reset to the new number of classes "
                f"(={new_num_classes})."
            )
            warnings.warn(msg, stacklevel=0)
            self._reset_prediction_layer(num_classes=label_info.num_classes)

        self._label_info = label_info

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
        # If customize_inputs is overridden
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

        for param_name, info in self.classification_layers.items():
            model_param = self.state_dict()[param_name].clone()
            ckpt_param = state_dict[prefix + param_name]
            stride = info.get("stride", 1)
            num_extra_classes = info.get("num_extra_classes", 0)
            for model_dst, ckpt_dst in enumerate(model2ckpt):
                if ckpt_dst >= 0:
                    model_param[(model_dst) * stride : (model_dst + 1) * stride].copy_(
                        ckpt_param[(ckpt_dst) * stride : (ckpt_dst + 1) * stride],
                    )
            if num_extra_classes > 0:
                num_ckpt_class = len(self.ckpt_classes)
                num_model_class = len(self.model_classes)
                model_param[(num_model_class) * stride : (num_model_class + 1) * stride].copy_(
                    ckpt_param[(num_ckpt_class) * stride : (num_ckpt_class + 1) * stride],
                )

            # Replace checkpoint weight by mixed weights
            state_dict[prefix + param_name] = model_param

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

    def export(self, output_dir: Path, export_format: OTXExportFormatType, input_size: Tuple[int, int],
               precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32, mean: Tuple[float, float,float] = (0., 0., 0.),
               std: Tuple[float, float,float] = (1., 1., 1.), resize_mode: str = "standard", pad_value: int = 0, swap_rgb: bool = False) -> None:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory path to save exported binary files.
            export_format: Format in which this `OTXModel` is exported.
        """

        metadata = self._generate_model_metadata(mean, std, resize_mode, pad_value, swap_rgb)

        if export_format == OTXExportFormatType.OPENVINO:
            self._export_to_openvino(output_dir, input_size, precision, metadata)
        if export_format == OTXExportFormatType.ONNX:
            self._export_to_onnx(output_dir, input_size, precision, metadata)
        if export_format == OTXExportFormatType.EXPORTABLE_CODE:
            self._export_to_exportable_code()

    def _generate_model_metadata(self, mean: Tuple[float, float,float],
                                 std: Tuple[float, float,float], resize_mode: str,
                                 pad_value: int, swap_rgb: bool) -> Dict[Tuple[str, str], Any]:
        """Embeds metadata to the exported model"""
        #raise NotImplementedError
        return {}

    @staticmethod
    def _embed_openvino_ir_metadata(ov_model: openvino.Model, metadata:  Dict[Tuple[str, str], Any]) -> openvino.Model:
        """Embeds metadata to OpenVINO model"""

        for k, data in metadata.items():
            ov_model.set_rt_info(data, list(k))

        return ov_model

    def _export_to_openvino(self, output_dir: Path, input_size: Tuple[int, int],
                            precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
                            metadata: Dict[Tuple[str, str], Any] = {}) -> None:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            output_dir: Directory path to save exported binary files
        """
        dummy_tensor = torch.rand((1, 3, *input_size)).to(next(self.model.parameters()).device)
        exported_model = openvino.convert_model(self.model, example_input=dummy_tensor)
        exported_model = OTXModel._embed_openvino_ir_metadata(exported_model, metadata)
        save_path = output_dir / (self._EXPORTED_MODEL_BASE_NAME + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXExportPrecisionType.FP16))
        raise NotImplementedError

    @staticmethod
    def _embed_onnx_metadata(onnx_model: onnx.ModelProto, metadata: Dict[Tuple[str, str], Any]) -> onnx.ModelProto:
        """Embeds metadata to ONNX model"""

        for item in metadata:
            meta = onnx_model.metadata_props.add()
            attr_path = " ".join(map(str, item))
            meta.key = attr_path.strip()
            meta.value = str(metadata[item])

        return onnx_model

    def _export_to_onnx(self, output_dir: Path, input_size: Tuple[int, int],
                        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
                        metadata: Dict[Tuple[str, str], Any] = {}) -> None:
        """Export to ONNX format.

        Args:
            output_dir: Directory path to save exported binary files
        """
        dummy_tensor = torch.rand((1, 3, *input_size)).to(next(self.model.parameters()).device)
        save_path = str(output_dir / (self._EXPORTED_MODEL_BASE_NAME + ".onnx"))
        torch.onnx.export(self.model, dummy_tensor, save_path)
        onnx_model = onnx.load(save_path)
        onnx_model = OTXModel._embed_onnx_metadata(onnx_model, metadata)
        if precision == OTXExportPrecisionType.FP16:
            from onnxconverter_common import float16
            onnx_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model, save_path)

    def _export_to_exportable_code(self) -> None:
        """Export to exportable code format.

        Args:
            output_dir: Directory path to save exported binary files
        """
        raise NotImplementedError

    def register_explain_hook(self) -> None:
        """Register explain hook.

        TBD
        """
        raise NotImplementedError

    def _reset_prediction_layer(self, num_classes: int) -> None:
        """Reset its prediction layer with a given number of classes.

        Args:
            num_classes: Number of classes
        """
        raise NotImplementedError
