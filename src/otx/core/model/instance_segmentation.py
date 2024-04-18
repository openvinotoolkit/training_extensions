# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for instance segmentation model entity used in OTX."""

from __future__ import annotations

import logging as log
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

import numpy as np
import torch
from mmengine.structures.instance_data import InstanceData
from openvino.model_api.models import Model
from openvino.model_api.tilers import InstanceSegmentationTiler
from torchvision import tv_tensors

from otx.algo.explain.explain_algo import get_feature_vector
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.tile import OTXTileBatchDataEntity
from otx.core.metrics import MetricInput
from otx.core.metrics.mean_ap import MaskRLEMeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.mask_util import encode_rle, polygon_to_rle
from otx.core.utils.tile_merge import InstanceSegTileMerge

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from mmdet.models.detectors.two_stage import TwoStageDetector
    from mmdet.structures import OptSampleList
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import InstanceSegmentationResult
    from torch import nn
    from torchmetrics import Metric

    from otx.core.metrics import MetricCallable


class OTXInstanceSegModel(OTXModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity]):
    """Base class for the Instance Segmentation models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self._tile_config = TileConfig()

    def forward_tiles(self, inputs: OTXTileBatchDataEntity[InstanceSegBatchDataEntity]) -> InstanceSegBatchPredEntity:
        """Unpack instance segmentation tiles.

        Args:
            inputs (TileBatchInstSegDataEntity): Tile batch data entity.

        Returns:
            InstanceSegBatchPredEntity: Merged instance segmentation prediction.
        """
        tile_preds: list[InstanceSegBatchPredEntity] = []
        tile_attrs: list[list[dict[str, int | str]]] = []
        merger = InstanceSegTileMerge(
            inputs.imgs_info,
            self.num_classes,
            self.tile_config,
        )
        for batch_tile_attrs, batch_tile_input in inputs.unbind():
            output = self.forward_explain(batch_tile_input) if self.explain_mode else self.forward(batch_tile_input)
            if isinstance(output, OTXBatchLossEntity):
                msg = "Loss output is not supported for tile merging"
                raise TypeError(msg)
            tile_preds.append(output)
            tile_attrs.append(batch_tile_attrs)
        pred_entities = merger.merge(tile_preds, tile_attrs)

        pred_entity = InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=[pred_entity.image for pred_entity in pred_entities],
            imgs_info=[pred_entity.img_info for pred_entity in pred_entities],
            scores=[pred_entity.score for pred_entity in pred_entities],
            bboxes=[pred_entity.bboxes for pred_entity in pred_entities],
            labels=[pred_entity.labels for pred_entity in pred_entities],
            masks=[pred_entity.masks for pred_entity in pred_entities],
            polygons=[pred_entity.polygons for pred_entity in pred_entities],
        )
        if self.explain_mode:
            pred_entity.saliency_map = [pred_entity.saliency_map for pred_entity in pred_entities]
            pred_entity.feature_vector = [pred_entity.feature_vector for pred_entity in pred_entities]

        return pred_entity

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="MaskRCNN",
            task_type="instance_segmentation",
            confidence_threshold=self.hparams.get("best_confidence_threshold", 0.0),
            iou_threshold=0.5,
            tile_config=self.tile_config if self.tile_config.enable_tiler else None,
        )

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state_dict from checkpoint.

        For detection, it is need to update confidence threshold information when
        the metric is FMeasure.
        """
        if best_confidence_threshold := ckpt.get("confidence_threshold", None) or (
            (hyper_parameters := ckpt.get("hyper_parameters", None))
            and (best_confidence_threshold := hyper_parameters.get("best_confidence_threshold", None))
        ):
            self.hparams["best_confidence_threshold"] = best_confidence_threshold
        super().load_state_dict(ckpt, *args, **kwargs)

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        if key == "val":
            retval = super()._log_metrics(meter, key)

            # NOTE: Validation metric logging can update `best_confidence_threshold`
            if best_confidence_threshold := getattr(meter, "best_confidence_threshold", None):
                self.hparams["best_confidence_threshold"] = best_confidence_threshold

            return retval

        if key == "test":
            # NOTE: Test metric logging should use `best_confidence_threshold` found previously.
            best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
            compute_kwargs = (
                {"best_confidence_threshold": best_confidence_threshold} if best_confidence_threshold else {}
            )

            return super()._log_metrics(meter, key, **compute_kwargs)

        raise ValueError(key)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: InstanceSegBatchPredEntity,
        inputs: InstanceSegBatchDataEntity,
    ) -> MetricInput:
        """Convert the prediction entity to the format that the metric can compute and cache the ground truth.

        This function will convert mask to RLE format and cache the ground truth for the current batch.

        Args:
            preds (InstanceSegBatchPredEntity): Current batch predictions.
            inputs (InstanceSegBatchDataEntity): Current batch ground-truth inputs.

        Returns:
            dict[str, list[dict[str, Tensor]]]: The converted predictions and ground truth.
        """
        pred_info = []
        target_info = []

        for bboxes, masks, scores, labels in zip(
            preds.bboxes,
            preds.masks,
            preds.scores,
            preds.labels,
        ):
            pred_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": [encode_rle(mask) for mask in masks.data],
                    "scores": scores,
                    "labels": labels,
                },
            )

        for imgs_info, bboxes, masks, polygons, labels in zip(
            inputs.imgs_info,
            inputs.bboxes,
            inputs.masks,
            inputs.polygons,
            inputs.labels,
        ):
            rles = (
                [encode_rle(mask) for mask in masks.data]
                if len(masks)
                else polygon_to_rle(polygons, *imgs_info.ori_shape)
            )
            target_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": rles,
                    "labels": labels,
                },
            )
        return {"preds": pred_info, "target": target_info}


class ExplainableOTXInstanceSegModel(OTXInstanceSegModel):
    """OTX Instance Segmentation model which can attach a XAI (Explainable AI) branch."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

        from otx.algo.explain.explain_algo import get_feature_vector

        self.model.feature_vector_fn = get_feature_vector
        self.model.explain_fn = self.get_explain_fn()

    def forward_explain(self, inputs: InstanceSegBatchDataEntity) -> InstanceSegBatchPredEntity:
        """Model forward function."""
        if isinstance(inputs, OTXTileBatchDataEntity):
            return self.forward_tiles(inputs)

        self.model.feature_vector_fn = get_feature_vector
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
        from otx.algo.explain.explain_algo import MaskRCNNExplainAlgo

        explainer = MaskRCNNExplainAlgo(num_classes=self.num_classes)
        return explainer.func

    @contextmanager
    def export_model_forward_context(self) -> Iterator[None]:
        """A context manager for managing the model's forward function during model exportation.

        It temporarily modifies the model's forward function to generate output sinks
        for explain results during the model graph tracing.
        """
        try:
            self._reset_model_forward()
            yield
        finally:
            self._restore_model_forward()

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


class MMDetInstanceSegCompatibleModel(ExplainableOTXInstanceSegModel):
    """Instance Segmentation model compatible for MMDet."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        config: DictConfig,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
    ) -> None:
        config = inplace_num_classes(cfg=config, num_classes=self._dispatch_label_info(label_info).num_classes)
        self.config = config
        self.load_from = self.config.pop("load_from", None)
        self.image_size: tuple[int, int, int, int] | None = None
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

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
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
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

            saliency_map = outputs["saliency_map"].detach().cpu().numpy()
            feature_vector = outputs["feature_vector"].detach().cpu().numpy()

            return InstanceSegBatchPredEntity(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                polygons=[],
                labels=labels,
                saliency_map=list(saliency_map),
                feature_vector=list(feature_vector),
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


class OVInstanceSegmentationModel(
    OVModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity],
):
    """Instance segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "MaskRCNN",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MaskRLEMeanAPCallable,
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

        if model_adapter.model.has_rt_info(["model_info", "confidence_threshold"]):
            best_confidence_threshold = model_adapter.model.get_rt_info(["model_info", "confidence_threshold"]).value
            self.hparams["best_confidence_threshold"] = best_confidence_threshold
        else:
            msg = (
                "Cannot get best_confidence_threshold from OpenVINO IR's rt_info. "
                "Please check whether this model is trained by OTX or not. "
                "Without this information, it can produce a wrong F1 metric score. "
                "At this time, it will be set as the default value = 0.0."
            )
            log.warning(msg)
            self.hparams["best_confidence_threshold"] = 0.0

        return Model.create_model(model_adapter, model_type=self.model_type, configuration=self.model_api_configuration)

    def _customize_outputs(
        self,
        outputs: list[InstanceSegmentationResult],
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
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
            return InstanceSegBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                polygons=[],
                labels=labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
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

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: InstanceSegBatchPredEntity,
        inputs: InstanceSegBatchDataEntity,
    ) -> MetricInput:
        """Convert the prediction entity to the format that the metric can compute and cache the ground truth.

        This function will convert mask to RLE format and cache the ground truth for the current batch.

        Args:
            preds (InstanceSegBatchPredEntity): Current batch predictions.
            inputs (InstanceSegBatchDataEntity): Current batch ground-truth inputs.

        Returns:
            dict[str, list[dict[str, Tensor]]]: The converted predictions and ground truth.
        """
        pred_info = []
        target_info = []

        for bboxes, masks, scores, labels in zip(
            preds.bboxes,
            preds.masks,
            preds.scores,
            preds.labels,
        ):
            pred_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": [encode_rle(mask) for mask in masks.data],
                    "scores": scores,
                    "labels": labels,
                },
            )

        for imgs_info, bboxes, masks, polygons, labels in zip(
            inputs.imgs_info,
            inputs.bboxes,
            inputs.masks,
            inputs.polygons,
            inputs.labels,
        ):
            rles = (
                [encode_rle(mask) for mask in masks.data]
                if len(masks)
                else polygon_to_rle(polygons, *imgs_info.ori_shape)
            )
            target_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": rles,
                    "labels": labels,
                },
            )
        return {"preds": pred_info, "target": target_info}

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", 0.0)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._log_metrics(meter, key, **compute_kwargs)
