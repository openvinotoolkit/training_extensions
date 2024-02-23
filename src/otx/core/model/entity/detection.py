# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torchvision import tv_tensors

from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity, DetBatchPredEntityWithXAI
from otx.core.data.entity.tile import TileBatchDetDataEntity
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.tile_merge import DetectionTileMerge
from otx.core.utils.utils import get_mean_std_from_data_processing

from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)

from typing import TYPE_CHECKING, Any, Callable
import types
from mmdet.structures import OptSampleList

from otx.algo.detection.heads.custom_ssd_head import CustomSSDHead
from otx.algo.hooks.recording_forward_hook import DetClassProbabilityMapHook


if TYPE_CHECKING:
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import DetectionResult
    from torch import nn

    from otx.core.exporter.base import OTXModelExporter


class OTXDetectionModel(
    OTXModel[DetBatchDataEntity, DetBatchPredEntity, DetBatchPredEntityWithXAI, TileBatchDetDataEntity],
):
    """Base class for the detection models used in OTX."""

    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.tile_config = TileConfig()

    def forward_tiles(self, inputs: TileBatchDetDataEntity) -> DetBatchPredEntity | DetBatchPredEntityWithXAI:
        """Unpack detection tiles.

        Args:
            inputs (TileBatchDetDataEntity): Tile batch data entity.

        Returns:
            DetBatchPredEntity: Merged detection prediction.
        """
        tile_preds: list[DetBatchPredEntity | DetBatchPredEntityWithXAI] = []
        tile_attrs: list[list[dict[str, int | str]]] = []
        merger = DetectionTileMerge(
            inputs.imgs_info,
            self.tile_config.iou_threshold,
            self.tile_config.max_num_instances,
        )
        for batch_tile_attrs, batch_tile_input in inputs.unbind():
            output = self.forward(batch_tile_input)
            if isinstance(output, OTXBatchLossEntity):
                msg = "Loss output is not supported for tile merging"
                raise TypeError(msg)
            tile_preds.append(output)
            tile_attrs.append(batch_tile_attrs)
        pred_entities = merger.merge(tile_preds, tile_attrs)

        return DetBatchPredEntity(
            batch_size=inputs.batch_size,
            images=[pred_entity.image for pred_entity in pred_entities],
            imgs_info=[pred_entity.img_info for pred_entity in pred_entities],
            scores=[pred_entity.score for pred_entity in pred_entities],
            bboxes=[pred_entity.bboxes for pred_entity in pred_entities],
            labels=[pred_entity.labels for pred_entity in pred_entities],
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        parameters["metadata"].update(
            {
                ("model_info", "model_type"): "ssd",
                ("model_info", "task_type"): "detection",
                ("model_info", "confidence_threshold"): str(0.0),  # it was able to be set in OTX 1.X
                ("model_info", "iou_threshold"): str(0.5),
            },
        )
        parameters["additional_output_names"] = ["saliency_map"] if self.explain_mode else []
        return parameters


class ExplainableOTXDetModel(OTXDetectionModel):
    """OTX detection model which can attach a XAI hook."""

    @torch.no_grad()
    def cls_head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward and returns cls scores.

        This can be redefined at the model's level.
        """
        if (head := getattr(self.model, "bbox_head", None)) is None:
            raise ValueError

        # if (neck := getattr(self.model, "neck", None)) is not None:
        #     x = neck(x)

        head_out = head(x)
        # Return the first output form detection head: classification scores
        return head_out[0]


    def forward_explain(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        self.model.explain_fn = self.get_explain_fn()

        # If customize_inputs is overridden
        outputs = (
            self._forward_explain_detection(self.model, **self._customize_inputs(inputs))
            if self._customize_inputs != ExplainableOTXDetModel._customize_inputs
            else self.model(inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != ExplainableOTXDetModel._customize_outputs
            else outputs
        )

    
    @staticmethod
    def _forward_explain_detection(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):
        """Forward func of the BaseDetector instance, which located in is in ExplainableOTXDetModel().model."""

        # Hack to remove grads for model parameters, since after class patching 
        # convolutions are failing since thay can't process gradients
        for param in self.parameters():
            param.requires_grad = False
        
        backbone_feat = self.extract_feat(inputs)
        bbox_head_feat = self.bbox_head.forward(backbone_feat)
    
        # Return the first output form detection head: classification scores
        saliency_map = self.explain_fn(bbox_head_feat[0])

        if mode == 'loss':
            predictions = self.bbox_head.loss(backbone_feat, data_samples)
        elif mode == 'predict':
            predictions = self.bbox_head.predict(backbone_feat, data_samples)
        elif mode == 'tensor':
            predictions = bbox_head_feat
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

        return {
            "predictions": predictions,
            "saliency_map": saliency_map,
        }


    def get_explain_fn(self) -> Callable:
        """Returns explain function. Lox"""

        # SSD-like heads also have background class
        background_class = isinstance(self.model.bbox_head, CustomSSDHead)
        explainer = DetClassProbabilityMapHook(
            self.cls_head_forward_fn,
            num_classes=self.num_classes + background_class,
            num_anchors=self.get_num_anchors(),
        )
        return explainer.func

    
    def _reset_model_forward(self) -> None:

        if not self.explain_mode:
            return

        self.model.explain_fn = self.get_explain_fn()
        forward_with_explain = self._forward_explain_detection

        self.original_model_forward = self.model.forward

        func_type = types.MethodType
        # Patch class method
        model_class = type(self.model)
        model_class.forward = func_type(forward_with_explain, self.model)

    def _restore_model_forward(self) -> None:
        if not self.explain_mode:
            return

        if not self.original_model_forward:
            msg = "Original model forward was not saved."
            raise RuntimeError(msg)

        func_type = types.MethodType
        self.model.forward = func_type(self.original_model_forward, self.model)
        self.original_model_forward = None


    def get_num_anchors(self) -> list[int]:
        """Gets the anchor configuration from model."""
        if anchor_generator := getattr(self.model.bbox_head, "prior_generator", None):
            return (
                anchor_generator.num_base_anchors
                if hasattr(anchor_generator, "num_base_anchors")
                else anchor_generator.num_base_priors
            )

        return [1] * 10


class MMDetCompatibleModel(ExplainableOTXDetModel):
    """Detection model compatible for MMDet.

    It can consume MMDet model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX detection model
    compatible for OTX pipelines.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size: tuple[int, int, int, int] | None = None
        super().__init__(num_classes=num_classes)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        if self.image_size is None:
            error_msg = "self.image_size shouldn't be None to use mmdeploy."
            raise ValueError(error_msg)

        export_params = super()._export_parameters
        export_params.update(get_mean_std_from_data_processing(self.config))
        export_params["model_builder"] = self._create_model
        export_params["model_cfg"] = copy.copy(self.config)
        export_params["test_pipeline"] = self._make_fake_test_pipeline()

        return export_params

    def _create_model(self) -> nn.Module:
        from .utils.mmdet import create_model

        model, self.classification_layers = create_model(self.config, self.load_from)
        return model

    def _make_fake_test_pipeline(self) -> list[dict[str, Any]]:
        return [
            {"type": "LoadImageFromFile"},
            {"type": "Resize", "scale": [self.image_size[3], self.image_size[2]], "keep_ratio": True},  # type: ignore[index]
            {"type": "LoadAnnotations", "with_bbox": True},
            {
                "type": "PackDetInputs",
                "meta_keys": ["ori_filenamescale_factor", "ori_shape", "filename", "img_shape", "pad_shape"],
            },
        ]

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        mmdet_inputs: dict[str, Any] = {}

        mmdet_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmdet_inputs["data_samples"] = [
            DetDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                    "ignored_labels": img_info.ignored_labels,
                },
                gt_instances=InstanceData(
                    bboxes=bboxes,
                    labels=labels,
                ),
            )
            for img_info, bboxes, labels in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
            )
        ]
        preprocessor: DetDataPreprocessor = self.model.data_preprocessor

        mmdet_inputs = preprocessor(data=mmdet_inputs, training=self.training)

        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | DetBatchPredEntityWithXAI | OTXBatchLossEntity:
        from mmdet.structures import DetDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, torch.Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores = []
        bboxes = []
        labels = []

        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for output in predictions:
            if not isinstance(output, DetDataSample):
                raise TypeError(output)
            scores.append(output.pred_instances.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    output.pred_instances.bboxes,
                    format="XYXY",
                    canvas_size=output.img_shape,
                ),
            )
            labels.append(output.pred_instances.labels)

        if self.explain_mode:
            if not isinstance(outputs, dict) or "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_maps = outputs["saliency_map"].detach().cpu().numpy()

            return DetBatchPredEntityWithXAI(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_maps=saliency_maps,
                feature_vectors=[],
            )

        return DetBatchPredEntity(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        from otx.core.exporter.mmdeploy import MMdeployExporter

        return MMdeployExporter(**self._export_parameters)


class OVDetectionModel(OVModel[DetBatchDataEntity, DetBatchPredEntity, DetBatchPredEntityWithXAI]):
    """Object detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_type: str = "SSD",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_classes,
            model_name,
            model_type,
            async_inference,
            max_num_requests,
            use_throughput_mode,
            model_api_configuration,
        )

    def _customize_outputs(
        self,
        outputs: list[DetectionResult],
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | DetBatchPredEntityWithXAI | OTXBatchLossEntity:
        # add label index
        bboxes = []
        scores = []
        labels = []
        for output in outputs:
            output_objects = output.objects
            if len(output_objects):
                bbox = [[output.xmin, output.ymin, output.xmax, output.ymax] for output in output_objects]
            else:
                bbox = torch.empty(size=(0, 0))
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    bbox,
                    format="XYXY",
                    canvas_size=inputs.imgs_info[-1].img_shape,
                ),
            )
            scores.append(torch.tensor([output.score for output in output_objects]))

            if self.model.get_label_name(0) == "background":
                # some OMZ model requires to shift labeles
                labels.append(torch.tensor([output.id - 1 for output in output_objects]))
            else:
                labels.append(torch.tensor([output.id for output in output_objects]))

        if outputs and outputs[0].saliency_map.size != 1:
            predicted_s_maps = [out.saliency_map for out in outputs]
            predicted_f_vectors = [out.feature_vector for out in outputs]

            return DetBatchPredEntityWithXAI(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_maps=predicted_s_maps,
                feature_vectors=predicted_f_vectors,
            )

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )
