# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base model entity used in OTX."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Iterable, Iterator, NamedTuple, Optional
import json

import numpy as np
import openvino
from attr import dataclass
from openvino.model_api.models import Model
from torch import nn

import nncf
from otx.core.data.dataset.base import LabelInfo
from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)
from otx.core.data.entity.tile import OTXTileBatchDataEntity, T_OTXTileBatchDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.utils.build import get_default_num_async_infer_requests
from jsonargparse import ArgumentParser
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from otx.core.data.module import OTXDataModule


class OTXModel(nn.Module, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity, T_OTXTileBatchDataEntity]):
    """Base class for the models used in OTX.

    Args:
        num_classes: Number of classes this model can predict.
    """

    _OPTIMIZED_MODEL_BASE_NAME: str = "optimized_model"
    _OPTIMIZE_DATASET_SIZE_LIMIT: int = 300

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

    @property
    def num_classes(self) -> int:
        """Returns model's number of classes. Can be redefined at the model's level."""
        return self.label_info.num_classes

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
        """Customize OTX output batch data entity if needed for model."""
        raise NotImplementedError

    def forward(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        # If customize_inputs is overridden
        if isinstance(inputs, OTXTileBatchDataEntity):
            return self.forward_tiles(inputs)

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

    def forward_tiles(self, inputs: T_OTXTileBatchDataEntity) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward function for tile task."""
        raise NotImplementedError

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
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model
        Returns:
            Path: path to the exported model.
        """
        return self._exporter.export(self.model, output_dir, base_name, export_format, precision)

    @property
    def _exporter(self) -> OTXModelExporter:
        msg = (
            "To export this OTXModel, you should implement an appropriate exporter for it. "
            "You can try to reuse ones provided in `otx.core.exporter.*`."
        )
        raise NotImplementedError(msg)

    @property
    def _export_parameters(self) -> dict[tuple[str, str], str]:
        """Defines parameters required to export a particular model implementation.
        Returns:
            dict[str, Any]: parameters of exporter.
        """
        parameters = {}

        all_labels = ""
        all_label_ids = ""
        for lbl in self.label_info.label_names:
            all_labels += lbl.replace(" ", "_") + " "
            all_label_ids += lbl.replace(" ", "_") + " "

        # not every model requires ptq_config
        ptq_config = self.ptq_config if hasattr(self, "ptq_config") else ""
        parameters["metadata"] = {
            ("model_info", "labels"): all_labels.strip(),
            ("model_info", "label_ids"): all_label_ids.strip(),
            ("model_info", "ptq_config"): json.dumps(ptq_config),
        }

        return parameters

    def optimize(self, output_dir: Path, data_module: OTXDataModule) -> Path:
        """Runs NNCF quantization on the passed data. Works only for OpenVINO models.

        Args:
            output_dir (Path): working directory to save the optimized model.
            data_module (OTXDataModule): dataset to for calibration of quantized layers.

        Returns:
            Path: path to the resulting optimized OpenVINO model.
        """
        raise NotImplementedError

    def _create_exporter(
        self,
    ) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        raise NotImplementedError

    def _generate_model_metadata(


    def _export_to_exportable_code(self) -> Path:
        """Export to exportable code format.

        Args:
            output_dir: Directory path to save exported binary files.

        Returns:
            Path: path to the exported model.
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
    OVModel supports synchronous as well as asynchronous inference type.

    Args:
        num_classes: Number of classes this model can predict.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_type: str,
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.async_inference = async_inference
        self.num_requests = max_num_requests if max_num_requests is not None else get_default_num_async_infer_requests()
        self.use_throughput_mode = use_throughput_mode
        self.model_api_configuration = model_api_configuration if model_api_configuration is not None else {}
        super().__init__(num_classes)

    def _create_model(self) -> Model:
        """Create a OV model with help of Model API."""
        from openvino.model_api.adapters import OpenvinoAdapter, create_core, get_user_config

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_adapter = OpenvinoAdapter(
            create_core(),
            self.model_name,
            max_num_requests=self.num_requests,
            plugin_config=plugin_config,
        )
        return Model.create_model(model_adapter, model_type=self.model_type, configuration=self.model_api_configuration)

    def _customize_inputs(self, entity: T_OTXBatchDataEntity) -> dict[str, Any]:
        # restore original numpy image
        images = [np.transpose(im.cpu().numpy(), (1, 2, 0)) for im in entity.images]
        return {"inputs": images}

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for model."""
        raise NotImplementedError

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

    def optimize(self, output_dir: Path, data_module: OTXDataModule) -> Path:
        """Runs NNCF quantization."""
        output_model_path = output_dir / (self._OPTIMIZED_MODEL_BASE_NAME + ".xml")

        def check_if_quantized(model: openvino.Model) -> bool:
            """Checks if OpenVINO model is already quantized."""
            nodes = model.get_ops()
            return any(op.get_type_name() == "FakeQuantize" for op in nodes)

        ov_model = openvino.Core().read_model(self.model_name)

        if check_if_quantized(ov_model):
            msg = "Model is already optimized by PTQ"
            raise RuntimeError(msg)

        def transform_fn(data_batch: T_OTXBatchDataEntity) -> np.array:
            np_data = self._customize_inputs(data_batch)
            image = np_data["inputs"][0]
            resized_image = self.model.resize(image, (self.model.w, self.model.h))
            resized_image = self.model.input_transform(resized_image)
            return self.model._change_layout(resized_image)  # noqa: SLF001

        train_dataset = data_module.train_dataloader()

        @dataclass
        class OptimizeDatasetWrapper(Iterable):
            """Restricts the length of the underlying dataset."""

            dataset: DataLoader
            max_size: int

            def __len__(self):
                return min(self.max_size, len(self.dataset))

            def __iter__(self) -> Iterator:
                return self.dataset.__iter__()

        if train_dataset.batch_size != 1:
            msg = "Optimization pipeline supports only batch size 1."
            raise RuntimeError(msg)

        quantization_dataset = nncf.Dataset(  # type: ignore[attr-defined]
            OptimizeDatasetWrapper(train_dataset, self._OPTIMIZE_DATASET_SIZE_LIMIT),
            transform_fn,
        )
        ptq_config = self._generate_ptq_config(ov_model)
        compressed_model = nncf.quantize(  # type: ignore[attr-defined]
            ov_model,
            quantization_dataset,
            **ptq_config,
        )

        openvino.save_model(compressed_model, output_model_path)

        return output_model_path

    def _generate_ptq_config(self, ov_model: Model) -> dict:
        from nncf import IgnoredScope
        from nncf.common.quantization.structs import QuantizationPreset
        from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

        initial_ptq_config = json.loads(ov_model.rt_info["model_info"]["ptq_config"].value)
        if not initial_ptq_config:
            return {}
        argparser = ArgumentParser()
        if "advanced_parameters" in initial_ptq_config:
            argparser.add_class_arguments(AdvancedQuantizationParameters, "advanced_parameters")
        if "preset" in initial_ptq_config:
            initial_ptq_config["preset"] = QuantizationPreset(initial_ptq_config["preset"])
            argparser.add_argument("--preset", type=QuantizationPreset)
        if "ignored_scope" in initial_ptq_config:
            argparser.add_class_arguments(IgnoredScope, "ignored_scope", as_positional=True)

        initial_ptq_config = argparser.parse_object(initial_ptq_config)

        return argparser.instantiate_classes(initial_ptq_config).as_dict()
