# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.metrics import MetricInput
from otx.core.metrics.dice import SegmCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfo, LabelInfoTypes, SegLabelInfo
from otx.core.utils.config import inplace_num_classes
import torch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.utils.utils import get_mean_std_from_data_processing


if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmseg.models.data_preprocessor import SegDataPreProcessor
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import ImageResultWithSoftPrediction
    from torch import Tensor, nn

    from otx.core.metrics import MetricCallable


class OTXSegmentationModel(OTXModel[SegBatchDataEntity, SegBatchPredEntity]):
    """Base class for the detection models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        pretrained_weights: str | None = None,
        backbone_configuration: dict[str, Any] = {},
        decode_head_configuration: dict[str, Any] = {},
        criterion_configuration: dict[str, Any] = {}
    ):

        self.backbone_configuration = backbone_configuration
        self.decode_head_configuration = decode_head_configuration
        self.criterion_configuration = criterion_configuration
        self.pretrained_weights = pretrained_weights
        self.mean = decode_head_configuration.get("mean",  [0.485, 0.456, 0.406])
        self.std = decode_head_configuration.get("std", [0.229, 0.224, 0.225])
        self.image_size = decode_head_configuration.get("image_size", (512,512))

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _customize_inputs(self, entity: SegBatchDataEntity) -> dict[str, Any]:
        masks = torch.stack(entity.masks).long()
        inputs = {"images": entity.images, "masks" : masks}
        return inputs

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=None,
        )

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SegBatchDataEntity,
    ) -> SegBatchPredEntity | OTXBatchLossEntity:

        if self.training:
            losses = OTXBatchLossEntity()
            losses["loss"] = outputs
            return losses

        masks = []

        for output in outputs:
            masks.append(output.argmax(dim=0))

        return SegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Segmentation",
            task_type="segmentation",
            return_soft_prediction=True,
            soft_threshold=0.5,
            blur_strength=-1,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: SegBatchPredEntity,
        inputs: SegBatchDataEntity,
    ) -> MetricInput:
        return [
            {
                "preds": pred_mask,
                "target": target_mask,
            }
            for pred_mask, target_mask in zip(preds.masks, inputs.masks)
        ]

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        if isinstance(label_info, int):
            return SegLabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, Sequence) and all(isinstance(name, str) for name in label_info):
            return SegLabelInfo(label_names=label_info, label_groups=[label_info])
        if isinstance(label_info, SegLabelInfo):
            return label_info

        raise TypeError(label_info)


class MMSegCompatibleModel(OTXSegmentationModel):
    """Segmentation model compatible for MMSeg.

    It can consume MMSeg model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX detection model
    compatible for OTX pipelines.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        config: DictConfig,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
    ) -> None:
        config = inplace_num_classes(cfg=config, num_classes=self._dispatch_label_info(label_info).num_classes)
        self.config = config
        self.load_from = self.config.pop("load_from", None)
        self.image_size = (1, 3, 544, 544)
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        from .utils.mmseg import create_model

        model, self.classification_layers = create_model(self.config, self.load_from)
        return model

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
                    "ignored_labels": img_info.ignored_labels,
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

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model(inputs=image, mode="tensor")


class OVSegmentationModel(OVModel[SegBatchDataEntity, SegBatchPredEntity]):
    """Semantic segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX segmentation model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Segmentation",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ImageResultWithSoftPrediction],
        inputs: SegBatchDataEntity,
    ) -> SegBatchPredEntity | OTXBatchLossEntity:
        if outputs and outputs[0].saliency_map.size != 1:
            predicted_s_maps = [out.saliency_map for out in outputs]
            predicted_f_vectors = [out.feature_vector for out in outputs]
            return SegBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=[],
                masks=[tv_tensors.Mask(mask.resultImage) for mask in outputs],
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return SegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=[tv_tensors.Mask(mask.resultImage) for mask in outputs],
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: SegBatchPredEntity,
        inputs: SegBatchDataEntity,
    ) -> MetricInput:
        return [
            {
                "preds": pred_mask,
                "target": target_mask,
            }
            for pred_mask, target_mask in zip(preds.masks, inputs.masks)
        ]

    def _create_label_info_from_ov_ir(self) -> SegLabelInfo:
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            label_info = json.loads(ov_model.get_rt_info(["model_info", "label_info"]).value)
            return SegLabelInfo(**label_info)

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)
