# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.types.export import OTXExportFormatType, OTXExportPrecisionType
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from pathlib import Path

    from mmseg.models.data_preprocessor import SegDataPreProcessor
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import ImageResultWithSoftPrediction
    from torch import device, nn


class OTXSegmentationModel(OTXModel[SegBatchDataEntity, SegBatchPredEntity]):
    """Base class for the segmentation models used in OTX."""

    def _generate_model_metadata(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        resize_mode: str,
        pad_value: int,
        swap_rgb: bool,
    ) -> dict[tuple[str, str], Any]:
        metadata = super()._generate_model_metadata(mean, std, resize_mode, pad_value, swap_rgb)
        metadata[("model_info", "model_type")] = "Segmentation"
        metadata[("model_info", "task_type")] = "segmentation"
        metadata[("model_info", "return_soft_prediction")] = str(True)
        metadata[("model_info", "soft_threshold")] = str(0.5)
        metadata[("model_info", "blur_strength")] = str(1)

        return metadata


def _get_export_params_from_seg_mmconfig(config: DictConfig) -> dict[str, Any]:
    return {
        "mean": config["data_preprocessor"]["mean"],
        "std": config["data_preprocessor"]["std"],
    }


class MMSegCompatibleModel(OTXSegmentationModel):
    """Segmentation model compatible for MMSeg.

    It can consume MMSeg model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX detection model
    compatible for OTX pipelines.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.export_params = _get_export_params_from_seg_mmconfig(config)
        self.load_from = self.config.pop("load_from", None)
        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        from mmengine.registry import MODELS as MMENGINE_MODELS
        from mmseg.models.data_preprocessor import SegDataPreProcessor as _SegDataPreProcessor
        from mmseg.registry import MODELS

        # NOTE: For the history of this monkey patching, please see
        # https://github.com/openvinotoolkit/training_extensions/issues/2743
        @MMENGINE_MODELS.register_module(force=True)
        class SegDataPreProcessor(_SegDataPreProcessor):
            @property
            def device(self) -> device:
                try:
                    buf = next(self.buffers())
                except StopIteration:
                    return super().device
                else:
                    return buf.device

        self.classification_layers = get_classification_layers(self.config, MODELS, "model.")
        return build_mm_model(self.config, MODELS, self.load_from)

    def _customize_inputs(self, entity: SegBatchDataEntity) -> dict[str, Any]:
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample

        mmseg_inputs: dict[str, Any] = {}
        mmseg_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmseg_inputs["data_samples"] = [
            SegDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_sem_seg=PixelData(
                    data=masks,
                ),
            )
            for img_info, masks in zip(
                entity.imgs_info,
                entity.masks,
            )
        ]
        preprocessor: SegDataPreProcessor = self.model.data_preprocessor

        mmseg_inputs = preprocessor(data=mmseg_inputs, training=self.training)
        mmseg_inputs["mode"] = "loss" if self.training else "predict"

        return mmseg_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SegBatchDataEntity,
    ) -> SegBatchPredEntity | OTXBatchLossEntity:
        from mmseg.structures import SegDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if "loss" in k:
                    losses[k] = v
            return losses

        masks = []

        for output in outputs:
            if not isinstance(output, SegDataSample):
                raise TypeError(output)
            masks.append(output.pred_sem_seg.data)

        return SegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
        )

    def _configure_export_parameters(self) -> None:
        self.export_params["resize_mode"] = "standard"
        self.export_params["pad_value"] = 0
        self.export_params["swap_rgb"] = False
        self.export_params["via_onnx"] = False
        self.export_params["input_size"] = (1, 3, 512, 512)
        self.export_params["onnx_export_configuration"] = None

    def export(
        self,
        output_dir: Path,
        export_format: OTXExportFormatType,
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
    ) -> None:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory path to save exported binary files.
            export_format: Format in which this `OTXModel` is exported.
            precision: Precision of the exported model.
        """
        self._configure_export_parameters()
        self._export(output_dir, export_format, precision=precision, **self.export_params)


class OVSegmentationModel(OVModel):
    """Semantic segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX segmentation model compatible for OTX testing pipeline.
    """

    def _customize_outputs(
        self,
        outputs: list[ImageResultWithSoftPrediction],
        inputs: SegBatchDataEntity,
    ) -> SegBatchPredEntity | OTXBatchLossEntity:
        # add label index

        return SegBatchPredEntity(
            batch_size=1,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=[tv_tensors.Mask(mask.resultImage) for mask in outputs],
        )
