# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import importlib
import tempfile
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, NamedTuple

import numpy as np
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
from otx.core.utils.build import get_default_num_async_infer_requests
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from omegaconf import DictConfig


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

    def export(
        self,
        output_dir: Path,
        export_format: OTXExportFormatType,
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        test_pipeline: list[dict] | None = None,
    ) -> None:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory path to save exported binary files.
            export_format: Format in which this `OTXModel` is exported.
            precision: Precision of the exported model.
            test_pipeline: Test data pipeline. It's necessary if using mmdeploy.
        """
        raise NotImplementedError

    def _export(
        self,
        output_dir: Path,
        export_format: OTXExportFormatType,
        input_size: tuple[int, ...],
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        resize_mode: str = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        via_onnx: bool = False,
        onnx_export_configuration: dict[str, Any] | None = None,
        mmdeploy_config: str | None = None,
        mm_model_config: dict | None = None,
        test_pipeline: list[dict] | None = None,
    ) -> None:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory path to save exported binary files.
            export_format: Format in which this `OTXModel` is exported.
        """
        metadata = self._generate_model_metadata(mean, std, resize_mode, pad_value, swap_rgb)

        if export_format == OTXExportFormatType.OPENVINO:
            self._export_to_openvino(
                output_dir,
                input_size,
                precision,
                metadata,
                via_onnx,
                onnx_export_configuration,
                mmdeploy_config,
                mm_model_config,
                test_pipeline,
            )
        elif export_format == OTXExportFormatType.ONNX:
            self._export_to_onnx(
                output_dir,
                input_size,
                precision,
                metadata,
                onnx_export_configuration,
                mmdeploy_config=mmdeploy_config,
                mm_model_config=mm_model_config,
                test_pipeline=test_pipeline,
            )
        elif export_format == OTXExportFormatType.EXPORTABLE_CODE:
            self._export_to_exportable_code()

    def _generate_model_metadata(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        resize_mode: str,
        pad_value: int,
        swap_rgb: bool,
    ) -> dict[tuple[str, str], Any]:
        """Embeds metadata to the exported model."""
        all_labels = ""
        all_label_ids = ""
        for lbl in self.label_info.label_names:
            all_labels += lbl.replace(" ", "_") + " "
            all_label_ids += lbl.replace(" ", "_") + " "

        mean_str = " ".join(map(str, mean))
        std_str = " ".join(map(str, std))

        return {
            ("model_info", "labels"): all_labels.strip(),
            ("model_info", "label_ids"): all_label_ids.strip(),
            ("model_info", "mean_values"): mean_str.strip(),
            ("model_info", "scale_values"): std_str.strip(),
            ("model_info", "resize_type"): resize_mode,
            ("model_info", "pad_value"): str(pad_value),
            ("model_info", "reverse_input_channels"): str(swap_rgb),
        }

    @staticmethod
    def _embed_openvino_ir_metadata(ov_model: openvino.Model, metadata: dict[tuple[str, str], Any]) -> openvino.Model:
        """Embeds metadata to OpenVINO model."""
        for k, data in metadata.items():
            ov_model.set_rt_info(data, list(k))

        return ov_model

    def _export_to_openvino(
        self,
        output_dir: Path,
        input_size: tuple[int, ...],
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], Any] | None = None,
        via_onnx: bool = False,
        onnx_configuration: dict[str, Any] | None = None,
        mmdeploy_config: str | None = None,
        mm_model_config: dict | None = None,
        test_pipeline: list[dict] | None = None,
    ) -> None:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            output_dir: Directory path to save exported binary files
        """
        dummy_tensor = torch.rand(input_size).to(next(self.model.parameters()).device)

        if mmdeploy_config is not None:
            from otx.core.model.utils.mmdeploy import MMdeployExporter
            from mmengine import Config as MMConfig
            config_module = importlib.import_module(mmdeploy_config)
            deploy_cfg = MMConfig.fromfile(config_module.__file__)

            # export ONNX
            exporter = MMdeployExporter(
                self._create_model,
                output_dir,
                mm_model_config,
                deploy_cfg,
                test_pipeline
            )
            onnx_path = exporter.cvt_torch2onnx()

            # export OpenVINO
            exported_model = openvino.convert_model(
                onnx_path,
                input=(openvino.runtime.PartialShape((1, 3, *input_size)),),
            )
            # tmp_ov_xml, tmp_ov_bin = exporter.cvt_onnx2openvino(onnx_path, precision)
            # ov_core = openvino.Core()
            # exported_model = ov_core.read_model(tmp_ov_xml, tmp_ov_bin)
            # os.remove(tmp_ov_xml)
            # os.remove(tmp_ov_bin)
        elif via_onnx:
            with tempfile.TemporaryDirectory() as tmpdirname:
                save_path = Path(tmpdirname)
                onnx_model_path = self._export_to_onnx(
                    save_path,
                    input_size,
                    precision,
                    metadata,
                    onnx_configuration,
                    True,
                )
                exported_model = openvino.convert_model(
                    onnx_model_path,
                    input=(openvino.runtime.PartialShape(input_size),),
                )
        else:
            exported_model = openvino.convert_model(
                self.model,
                example_input=dummy_tensor,
                input=(openvino.runtime.PartialShape(input_size),),
            )

        if metadata is None:
            metadata = {}
        exported_model = OTXModel._embed_openvino_ir_metadata(exported_model, metadata)
        save_path = output_dir / (self._EXPORTED_MODEL_BASE_NAME + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXExportPrecisionType.FP16))

    @staticmethod
    def _embed_onnx_metadata(onnx_model: onnx.ModelProto, metadata: dict[tuple[str, str], Any]) -> onnx.ModelProto:
        """Embeds metadata to ONNX model."""
        for item in metadata:
            meta = onnx_model.metadata_props.add()
            attr_path = " ".join(map(str, item))
            meta.key = attr_path.strip()
            meta.value = str(metadata[item])

        return onnx_model

    def _export_to_onnx(
        self,
        output_dir: Path,
        input_size: tuple[int, ...],
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], Any] | None = None,
        onnx_configuration: dict[str, Any] | None = None,
        return_path_to_export_ir: bool = False,
        mmdeploy_config: str | None = None,
        mm_model_config: dict | None = None,
        test_pipeline: list[dict] | None = None,
    ) -> None | str:
        """Export to ONNX format.

        Args:
            output_dir: Directory path to save exported binary files
        """
        dummy_tensor = torch.rand(input_size).to(next(self.model.parameters()).device)
        save_path = str(output_dir / (self._EXPORTED_MODEL_BASE_NAME + ".onnx"))
        export_configuration = onnx_configuration if onnx_configuration else {}

        torch.onnx.export(self.model, dummy_tensor, save_path, **export_configuration)
        if return_path_to_export_ir:
            return save_path

        onnx_model = onnx.load(save_path)
        if metadata is None:
            metadata = {}
        onnx_model = OTXModel._embed_onnx_metadata(onnx_model, metadata)
        if precision == OTXExportPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model, save_path)
        return None

    def need_mmdeploy(self):
        """Whether mmdeploy is used when exporting a model."""
        return False

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


class OVModel(OTXModel, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """Base class for the OpenVINO model.

    This is a base class representing interface for interacting with OpenVINO
    Intermidiate Representation (IR) models. OVModel can create and validate
    OpenVINO IR model directly from provided path locally or from
    OpenVINO OMZ repository. (Only PyTorch models are supported).
    OVModel supports synchoronous as well as asynchronous inference type.

    Args:
        num_classes: Number of classes this model can predict.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.model_name = config.pop("model_name")
        self.model_type = config.pop("model_type")
        self.async_inference = config.pop("async_inference", False)
        self.num_requests = config.pop("max_num_requests", get_default_num_async_infer_requests())
        self.use_throughput_mode = config.pop("use_throughput_mode", False)
        self.config = config
        super().__init__(num_classes)

    def _create_model(self) -> nn.Module:
        """Create a OV model with help of Model API."""
        from openvino.model_api.adapters import OpenvinoAdapter, create_core, get_user_config
        from openvino.model_api.models import Model

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
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

        def _callback(result: NamedTuple, idx: int) -> None:
            output_dict[idx] = result

        numpy_inputs = self._customize_inputs(inputs)["inputs"]
        if self.async_inference:
            output_dict: dict[int, NamedTuple] = {}
            self.model.set_callback(_callback)
            for idx, im in enumerate(numpy_inputs):
                if not self.model.is_ready():
                    self.model.await_any()
                self.model.infer_async(im, user_data=idx)
            self.model.await_all()
            outputs = [out[1] for out in sorted(output_dict.items())]
        else:
            outputs = [self.model(im) for im in numpy_inputs]

        return self._customize_outputs(outputs, inputs)
