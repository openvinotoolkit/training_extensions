# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for instance segmentation model entity used in OTX."""

from __future__ import annotations

import logging as log
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

import numpy as np
import torch
from model_api.tilers import InstanceSegmentationTiler
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchvision import tv_tensors

from otx.algo.explain.explain_algo import InstSegExplainAlgo, feature_vector_fn
from otx.algo.instance_segmentation.two_stage import TwoStageDetector
from otx.algo.utils.mmengine_utils import InstanceData, load_checkpoint
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.tile import OTXTileBatchDataEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.metrics import MetricInput
from otx.core.metrics.fmeasure import FMeasure
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.mask_util import encode_rle, polygon_to_rle
from otx.core.utils.tile_merge import InstanceSegTileMerge

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import InstanceSegmentationResult
    from torch import nn

    from otx.core.metrics import MetricCallable


class OTXInstanceSegModel(OTXModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity]):
    """Base class for the Instance Segmentation models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )
        self.input_size: tuple[int, int]

    def _build_model(self, num_classes: int) -> nn.Module:
        raise NotImplementedError

    def _create_model(self) -> nn.Module:
        detector = self._build_model(num_classes=self.label_info.num_classes)
        detector.init_weights()
        self.classification_layers = self.get_classification_layers("model.")

        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)
        inputs: dict[str, Any] = {}

        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"

        return inputs

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,  # TODO (sungchul): Remove `InstanceData`.
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # pop acc from losses
            losses.pop("acc", None)
            return losses

        scores: list[Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
            scores.append(prediction.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
            output_masks = tv_tensors.Mask(
                prediction.masks,
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(prediction.labels)

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

    def get_classification_layers(self, prefix: str = "") -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            config (DictConfig): Config for building model.
            model_registry (Registry): Registry for building model.
            prefix (str): Prefix of model param name.
                Normally it is "model." since OTXModel set it's nn.Module model as self.model

        Return:
            dict[str, dict[str, int]]
            A dictionary contain classification layer's name and information.
            Stride means dimension of each classes, normally stride is 1, but sometimes it can be 4
            if the layer is related bbox regression for object detection.
            Extra classes is default class except class from data.
            Normally it is related with background classes.
        """
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

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
            self.explain_mode,
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

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="MaskRCNN",
            task_type="instance_segmentation",
            confidence_threshold=self.hparams.get("best_confidence_threshold", 0.05),
            iou_threshold=0.5,
            tile_config=self.tile_config if self.tile_config.enable_tiler else None,
        )

    def on_load_checkpoint(self, ckpt: dict[str, Any]) -> None:
        """Load state_dict from checkpoint.

        For detection, it is need to update confidence threshold information when
        the metric is FMeasure.
        """
        if best_confidence_threshold := ckpt.get("confidence_threshold", None) or (
            (hyper_parameters := ckpt.get("hyper_parameters", None))
            and (best_confidence_threshold := hyper_parameters.get("best_confidence_threshold", None))
        ):
            self.hparams["best_confidence_threshold"] = best_confidence_threshold
        super().on_load_checkpoint(ckpt)

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        if key == "val":
            retval = super()._log_metrics(meter, key)

            # NOTE: Validation metric logging can update `best_confidence_threshold`
            if (
                isinstance(meter, MetricCollection)
                and (fmeasure := getattr(meter, "FMeasure", None))
                and (best_confidence_threshold := getattr(fmeasure, "best_confidence_threshold", None))
            ) or (
                isinstance(meter, FMeasure)
                and (best_confidence_threshold := getattr(meter, "best_confidence_threshold", None))
            ):
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

    def get_dummy_input(self, batch_size: int = 1) -> InstanceSegBatchDataEntity:
        """Returns a dummy input for instance segmentation model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        images = [torch.rand(3, *self.input_size) for _ in range(batch_size)]
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return InstanceSegBatchDataEntity(batch_size, images, infos, bboxes=[], masks=[], labels=[], polygons=[])


class ExplainableOTXInstanceSegModel(OTXInstanceSegModel):
    """OTX Instance Segmentation model which can attach a XAI (Explainable AI) branch."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()
        self.model.get_results_from_head = self.get_results_from_head

    def forward_explain(self, inputs: InstanceSegBatchDataEntity) -> InstanceSegBatchPredEntity:
        """Model forward function."""
        if isinstance(inputs, OTXTileBatchDataEntity):
            return self.forward_tiles(inputs)

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
    @torch.no_grad()
    def _forward_explain_inst_seg(
        self: TwoStageDetector,
        entity: InstanceSegBatchDataEntity,
        mode: str = "tensor",  # noqa: ARG004
    ) -> dict[str, Tensor]:
        """Forward func of the BaseDetector instance, which located in is in ExplainableOTXInstanceSegModel().model."""
        x = self.extract_feat(entity.images)

        feature_vector = self.feature_vector_fn(x)
        predictions = self.get_results_from_head(x, entity)

        if isinstance(predictions, tuple) and isinstance(predictions[0], Tensor):
            # Export case, consists of tensors
            # For OV task saliency map are generated on MAPI side
            saliency_map = torch.empty(1, dtype=torch.uint8)
        elif isinstance(predictions, list) and isinstance(predictions[0], InstanceData):
            # Predict case, consists of InstanceData
            saliency_map = self.explain_fn(predictions)
        else:
            msg = f"Unexpected predictions type: {type(predictions)}"
            raise TypeError(msg)

        return {
            "predictions": predictions,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

    def get_results_from_head(
        self,
        x: tuple[Tensor],
        entity: InstanceSegBatchDataEntity,
    ) -> tuple[Tensor] | list[InstanceData]:
        """Get the results from the head of the instance segmentation model.

        Args:
            x (tuple[Tensor]): The features from backbone and neck.
            data_samples (OptSampleList | None): A list of data samples.

        Returns:
            tuple[Tensor] | list[InstanceData]: The predicted results from the head of the model.
            Tuple for the Export case, list for the Predict case.
        """
        from otx.algo.instance_segmentation.rtmdet_inst import RTMDetInstTiny

        if isinstance(self, RTMDetInstTiny):
            return self.model.bbox_head.predict(x, entity, rescale=False)
        rpn_results_list = self.model.rpn_head.predict(x, entity, rescale=False)
        return self.model.roi_head.predict(x, rpn_results_list, entity, rescale=True)

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        explainer = InstSegExplainAlgo(num_classes=self.num_classes)
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
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Reads model configuration from ModelAPI OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): target adapter to read the config
        """
        if model_adapter.model.has_rt_info(["model_info", "confidence_threshold"]):
            best_confidence_threshold = model_adapter.model.get_rt_info(["model_info", "confidence_threshold"]).value
            self.hparams["best_confidence_threshold"] = float(best_confidence_threshold)
        else:
            msg = (
                "Cannot get best_confidence_threshold from OpenVINO IR's rt_info. "
                "Please check whether this model is trained by OTX or not. "
                "Without this information, it can produce a wrong F1 metric score. "
                "At this time, it will be set as the default value = None."
            )
            log.warning(msg)
            self.hparams["best_confidence_threshold"] = None

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
                    device=self.device,
                ),
            )
            # NOTE: OTX 1.5 filter predictions with result_based_confidence_threshold,
            # but OTX 2.0 doesn't have it in configuration.
            _masks = [output.mask for output in output_objects]
            _masks = np.stack(_masks) if len(_masks) else []
            scores.append(torch.tensor([output.score for output in output_objects], device=self.device))
            masks.append(torch.tensor(_masks, device=self.device))
            labels.append(torch.tensor([output.id - 1 for output in output_objects], device=self.device))

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
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._log_metrics(meter, key, **compute_kwargs)
