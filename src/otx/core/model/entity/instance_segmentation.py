# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for instance segmentation model entity used in OTX."""

from __future__ import annotations

import json
import logging as log
import types
from copy import copy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from mmengine.structures.instance_data import InstanceData
from openvino.model_api.models import Model
from openvino.model_api.tilers import InstanceSegmentationTiler
from torchvision import tv_tensors

from otx.algo.hooks.recording_forward_hook import MaskRCNNRecordingForwardHook, feature_vector_fn
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import (
    OTXBatchLossEntity,
)
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchDataEntity,
    InstanceSegBatchPredEntity,
    InstanceSegBatchPredEntityWithXAI,
)
from otx.core.data.entity.tile import TileBatchInstSegDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.tile_merge import InstanceSegTileMerge
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from mmdet.models.detectors.base import TwoStageDetector
    from mmdet.structures import OptSampleList
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import InstanceSegmentationResult
    from torch import nn


class OTXInstanceSegModel(
    OTXModel[
        InstanceSegBatchDataEntity,
        InstanceSegBatchPredEntity,
        InstanceSegBatchPredEntityWithXAI,
        TileBatchInstSegDataEntity,
    ],
):
    """Base class for the Instance Segmentation models used in OTX."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tile_config = TileConfig()
        self.test_meta_info: dict[str, Any] = {}

    def forward_tiles(self, inputs: TileBatchInstSegDataEntity) -> InstanceSegBatchPredEntity:
        """Unpack instance segmentation tiles.

        Args:
            inputs (TileBatchInstSegDataEntity): Tile batch data entity.

        Returns:
            InstanceSegBatchPredEntity: Merged instance segmentation prediction.
        """
        tile_preds: list[InstanceSegBatchPredEntity | InstanceSegBatchPredEntityWithXAI] = []
        tile_attrs: list[list[dict[str, int | str]]] = []
        merger = InstanceSegTileMerge(
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

        return InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=[pred_entity.image for pred_entity in pred_entities],
            imgs_info=[pred_entity.img_info for pred_entity in pred_entities],
            scores=[pred_entity.score for pred_entity in pred_entities],
            bboxes=[pred_entity.bboxes for pred_entity in pred_entities],
            labels=[pred_entity.labels for pred_entity in pred_entities],
            masks=[pred_entity.masks for pred_entity in pred_entities],
            polygons=[pred_entity.polygons for pred_entity in pred_entities],
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        parameters["metadata"].update(
            {
                ("model_info", "model_type"): "MaskRCNN",
                ("model_info", "task_type"): "instance_segmentation",
                ("model_info", "confidence_threshold"): str(0.0),  # it was able to be set in OTX 1.X
                ("model_info", "iou_threshold"): str(0.5),
                ("model_info", "test_meta_info"): json.dumps(self.test_meta_info),
            },
        )

        # Instance segmentation needs to add empty label
        all_labels = "otx_empty_lbl "
        all_label_ids = "None "
        for lbl in self.label_info.label_names:
            all_labels += lbl.replace(" ", "_") + " "
            all_label_ids += lbl.replace(" ", "_") + " "

        parameters["metadata"][("model_info", "labels")] = all_labels.strip()
        parameters["metadata"][("model_info", "label_ids")] = all_label_ids.strip()

        if self.tile_config.enable_tiler:
            parameters["metadata"].update(
                {
                    ("model_info", "tile_size"): str(self.tile_config.tile_size[0]),
                    ("model_info", "tiles_overlap"): str(self.tile_config.overlap),
                    ("model_info", "max_pred_number"): str(self.tile_config.max_num_instances),
                },
            )

        return parameters


class ExplainableOTXInstanceSegModel(OTXInstanceSegModel):
    """OTX Instance Segmentation model which can attach a XAI hook."""

    def forward_explain(
        self,
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | InstanceSegBatchPredEntityWithXAI | OTXBatchLossEntity:
        """Model forward function."""
        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()

        # If customize_inputs is overridden
        outputs = (
            self._forward_explain_inst_seg(self.model, **self._customize_inputs(inputs))
            if self._customize_inputs != ExplainableOTXInstanceSegModel._customize_inputs
            else self._forward_explain_inst_seg(self.model, inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != ExplainableOTXInstanceSegModel._customize_outputs
            else outputs["predictions"]
        )

    @staticmethod
    def _forward_explain_inst_seg(
        self: TwoStageDetector,
        inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = "tensor",  # noqa: ARG004
    ) -> dict[str, torch.Tensor]:
        """Forward func of the BaseDetector instance, which located in is in ExplainableOTXInstanceSegModel().model."""
        # Workaround to remove grads for model parameters, since after class patching
        # convolutions are failing since thay can't process gradients
        for param in self.parameters():
            param.requires_grad = False

        x = self.extract_feat(inputs)

        feature_vector = self.feature_vector_fn(x)

        rpn_results_list = self.rpn_head.predict(x, data_samples, rescale=False)
        results_list = self.roi_head.predict(x, rpn_results_list, data_samples, rescale=True)

        if isinstance(results_list, tuple) and isinstance(results_list[0], torch.Tensor):  # rewrite
            # Export case, consists of tensors
            predictions = results_list
            # For OV task saliency map are generated on MAPI side
            saliency_map = torch.empty(1, dtype=torch.uint8)

        elif isinstance(results_list, list) and isinstance(results_list[0], InstanceData):  # rewrite
            # Predict case, consists of InstanceData
            predictions = self.add_pred_to_datasample(data_samples, results_list)

            features_for_sal_map = [data_sample.pred_instances for data_sample in data_samples]
            saliency_map = self.explain_fn(features_for_sal_map)

        return {
            "predictions": predictions,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        explainer = MaskRCNNRecordingForwardHook(num_classes=self.num_classes)
        return explainer.func

    def _reset_model_forward(self) -> None:
        if not self.explain_mode:
            return

        self.model.explain_fn = self.get_explain_fn()
        forward_with_explain = self._forward_explain_inst_seg

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

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        parameters["output_names"] = ["feature_vector", "saliency_map"] if self.explain_mode else None
        return parameters


class MMDetInstanceSegCompatibleModel(ExplainableOTXInstanceSegModel):
    """Instance Segmentation model compatible for MMDet."""

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = self.config.pop("load_from", None)
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
        export_params["model_cfg"] = copy(self.config)
        export_params["test_pipeline"] = self._make_fake_test_pipeline()

        return export_params

    def _create_model(self) -> nn.Module:
        from .utils.mmdet import create_model

        model, self.classification_layers = create_model(self.config, self.load_from)
        return model

    def _make_fake_test_pipeline(self) -> list[dict[str, Any]]:
        return [
            {"type": "LoadImageFromFile", "backend_args": None},
            {"type": "Resize", "scale": [self.image_size[3], self.image_size[2]], "keep_ratio": True},  # type: ignore[index]
            {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
            {
                "type": "PackDetInputs",
                "meta_keys": ["img_idimg_path", "ori_shape", "img_shape", "scale_factor"],
            },
        ]

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        from mmdet.structures import DetDataSample
        from mmdet.structures.mask import BitmapMasks, PolygonMasks
        from mmengine.structures import InstanceData

        mmdet_inputs: dict[str, Any] = {}

        mmdet_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmdet_inputs["data_samples"] = []

        for img_info, bboxes, masks, polygons, labels in zip(
            entity.imgs_info,
            entity.bboxes,
            entity.masks,
            entity.polygons,
            entity.labels,
        ):
            # NOTE: ground-truth masks are resized in training, but not in inference
            height, width = img_info.img_shape if self.training else img_info.ori_shape
            if len(masks):
                mmdet_masks = BitmapMasks(masks.data.cpu().numpy(), height, width)
            else:
                mmdet_masks = PolygonMasks(
                    [[np.array(polygon.points)] for polygon in polygons],
                    height,
                    width,
                )

            data_sample = DetDataSample(
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
                    masks=mmdet_masks,
                    labels=labels,
                ),
            )
            mmdet_inputs["data_samples"].append(data_sample)

        preprocessor: DetDataPreprocessor = self.model.data_preprocessor

        mmdet_inputs = preprocessor(data=mmdet_inputs, training=self.training)

        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs

    def _customize_outputs(
        self,
        outputs: dict[str, Any],
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | InstanceSegBatchPredEntityWithXAI | OTXBatchLossEntity:
        from mmdet.structures import DetDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, torch.Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            return losses

        scores: list[torch.Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for output in predictions:
            if not isinstance(output, DetDataSample):
                raise TypeError(output)

            scores.append(output.pred_instances.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    output.pred_instances.bboxes,
                    format="XYXY",
                    canvas_size=output.ori_shape,
                ),
            )
            output_masks = tv_tensors.Mask(
                output.pred_instances.masks,
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(output.pred_instances.labels)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_maps = outputs["saliency_map"].detach().cpu().numpy()
            feature_vectors = outputs["feature_vector"].detach().cpu().numpy()

            return InstanceSegBatchPredEntityWithXAI(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                polygons=[],
                labels=labels,
                saliency_maps=list(saliency_maps),
                feature_vectors=list(feature_vectors),
            )

        return InstanceSegBatchPredEntity(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        from otx.core.exporter.mmdeploy import MMdeployExporter

        return MMdeployExporter(**self._export_parameters)


class OVInstanceSegmentationModel(
    OVModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity, InstanceSegBatchPredEntityWithXAI],
):
    """Instance segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_type: str = "MaskRCNN",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
    ) -> None:
        self.test_meta_info: dict[str, Any] = {}
        super().__init__(
            num_classes,
            model_name,
            model_type,
            async_inference,
            max_num_requests,
            use_throughput_mode,
            model_api_configuration,
        )

    def _setup_tiler(self) -> None:
        """Setup tiler for tile task."""
        execution_mode = "async" if self.async_inference else "sync"
        # Note: Disable async_inference as tiling has its own sync/async implementation
        self.async_inference = False
        self.model = InstanceSegmentationTiler(self.model, execution_mode=execution_mode)
        log.info(
            f"Enable tiler with tile size: {self.model.tile_size} \
                and overlap: {self.model.tiles_overlap}",
        )

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
            model_parameters=self.model_adapter_parameters,
        )
        for name, info in model_adapter.model.rt_info["model_info"].items():
            if name == "test_meta_info":
                for key, value in json.loads(info.value).items():
                    self.test_meta_info[key] = value
        return Model.create_model(model_adapter, model_type=self.model_type, configuration=self.model_api_configuration)

    def _customize_outputs(
        self,
        outputs: list[InstanceSegmentationResult],
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | InstanceSegBatchPredEntityWithXAI | OTXBatchLossEntity:
        # add label index
        bboxes = []
        scores = []
        labels = []
        masks = []
        for output in outputs:
            output_objects = output.segmentedObjects
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
            # NOTE: OTX 1.5 filter predictions with result_based_confidence_threshold,
            # but OTX 2.0 doesn't have it in configuration.
            _masks = [output.mask for output in output_objects]
            _masks = np.stack(_masks) if len(_masks) else []
            scores.append(torch.tensor([output.score for output in output_objects]))
            masks.append(torch.tensor(_masks))
            labels.append(torch.tensor([output.id - 1 for output in output_objects]))

        if outputs and outputs[0].saliency_map:
            predicted_s_maps = []
            for out in outputs:
                image_map = np.array([s_map for s_map in out.saliency_map if s_map.ndim > 1])
                predicted_s_maps.append(image_map)

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return InstanceSegBatchPredEntityWithXAI(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                polygons=[],
                labels=labels,
                saliency_maps=predicted_s_maps,
                feature_vectors=predicted_f_vectors,
            )

        return InstanceSegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )
