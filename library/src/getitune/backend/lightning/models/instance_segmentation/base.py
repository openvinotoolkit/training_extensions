# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for instance segmentation model entity used in getitune."""

# type: ignore[override]

from __future__ import annotations

import copy
import logging as log
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, Literal, Sequence, cast

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchvision import tv_tensors
from torchvision.models.detection.image_list import ImageList

from getitune.backend.lightning.models.base import (
    DataInputParams,
    DefaultOptimizerCallable,
    DefaultSchedulerCallable,
    LightningModel,
)
from getitune.backend.lightning.models.common.pretrained_weights import PretrainedWeightsMixin
from getitune.backend.lightning.models.common.target_utils import align_sample_batch_annotations
from getitune.backend.lightning.models.instance_segmentation.segmentors.maskrcnn_tv import MaskRCNN
from getitune.backend.lightning.models.instance_segmentation.segmentors.two_stage import TwoStageDetector
from getitune.backend.lightning.models.utils.utils import InstanceData
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.backend.lightning.tools.explain.explain_algo import InstSegExplainAlgo, feature_vector_fn
from getitune.backend.lightning.tools.tile_merge import InstanceSegTileMerge
from getitune.config.data import TileConfig
from getitune.data.entity.base import BatchLoss, ImageInfo
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.data.entity.tile import TileBatchData
from getitune.data.entity.utils import stack_batch
from getitune.data.utils.structures.mask.mask_util import encode_rle
from getitune.metrics import MetricInput
from getitune.metrics.fmeasure import FMeasure, MaskRLEMeanAPFMeasureCallable
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfoTypes
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from datumaro.experimental.fields import TileInfo
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable
    from getitune.types import PathLike


class LightningInstanceSegModel(PretrainedWeightsMixin, LightningModel):
    """Base class for the Instance Segmentation models used in getitune.

    NOTE: LightningInstanceSegModel has many duplicate methods to LightningDetectionModel,
    however, it is not a subclass of LightningDetectionModel because it has different
    export parameters and different metric computation. Some refactor could be done
    to reduce the code duplication in the future.

    Args:
        label_info (LabelInfoTypes | int | Sequence): Information about the labels used in the model.
            If `int` is given, label info will be constructed from number of classes,
            if `Sequence` is given, label info will be constructed from the sequence of label names.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
            If None is given, default parameters for the specific model will be used.
        model_name (str, optional): Name of the model. Defaults to "inst_segm_model".
        optimizer (OptimizerCallable, optional): Optimizer for the model. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Scheduler for the model.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for evaluating the model.
            Defaults to MaskRLEMeanAPFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    pretrained_urls: ClassVar[dict[str, str]]
    _nms_always_embedded: ClassVar[bool] = False

    def __init__(
        self,
        label_info: LabelInfoTypes | int | Sequence,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "inst_segm_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
        )

        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()
        self.model.get_results_from_head = self.get_results_from_head
        self.export_nms = export_nms

    def _customize_inputs(self, entity: SampleBatch) -> dict[str, Any]:
        # Defensively realign per-image boxes/labels/masks counts so a divergence
        # (e.g. from tiling or crop augmentations) does not crash mmdet's
        # InstanceData, which requires all instance-level fields to share length.
        if self.training:
            align_sample_batch_annotations(entity)
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)  # type: ignore[assignment,arg-type]
        inputs: dict[str, Any] = {}

        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"

        return inputs

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = BatchLoss()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            losses.pop("acc", None)
            return losses

        scores: list[Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):  # type: ignore[arg-type]
            scores.append(prediction.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,
                    format="XYXY",
                    canvas_size=img_info.ori_shape,  # type: ignore[union-attr]
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

            return PredictionBatch(
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                labels=labels,
                saliency_map=list(saliency_map),
                feature_vector=list(feature_vector),
            )

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            labels=labels,
        )

    def forward_tiles(self, inputs: TileBatchData) -> PredictionBatch:
        """Unpack instance segmentation tiles.

        Args:
            inputs (TileBatchData): Tile batch data entity.

        Returns:
            TorchPredBatch: Merged instance segmentation prediction.
        """
        tile_preds: list[PredictionBatch] = []
        tile_infos: list[list[TileInfo]] = []
        merger = InstanceSegTileMerge(
            inputs.imgs_info,
            self.num_classes,
            self.tile_config,
            self.explain_mode,
        )
        for batch_tile_infos, batch_tile_input in inputs.unbind(self.tile_config.tile_inference_batch_size):
            output = self.forward_explain(batch_tile_input) if self.explain_mode else self.forward(batch_tile_input)
            if isinstance(output, BatchLoss):
                msg = "Loss output is not supported for tile merging"
                raise TypeError(msg)
            tile_preds.append(output)
            tile_infos.append(batch_tile_infos)
        pred_entities = merger.merge(tile_preds, tile_infos)

        pred_entity = PredictionBatch(
            images=[pred_entity.image for pred_entity in pred_entities],
            imgs_info=[pred_entity.img_info for pred_entity in pred_entities],
            scores=[pred_entity.scores for pred_entity in pred_entities],
            bboxes=[pred_entity.bboxes for pred_entity in pred_entities],
            labels=[pred_entity.label for pred_entity in pred_entities],
            masks=[pred_entity.masks for pred_entity in pred_entities],
        )
        if self.explain_mode:
            pred_entity.saliency_map = [pred_entity.saliency_map for pred_entity in pred_entities]
            pred_entity.feature_vector = [pred_entity.feature_vector for pred_entity in pred_entities]

        return pred_entity

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...] | dict[str, Any]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(
            inputs,
            meta_info_list,
            explain_mode=self.explain_mode,
            with_nms=self._nms_always_embedded or self.export_nms,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        modified_label_info = copy.deepcopy(self.label_info)
        # Instance segmentation needs to add empty label to satisfy MAPI wrapper requirements
        modified_label_info.label_names.insert(0, "getitune_empty_lbl")
        modified_label_info.label_ids.insert(0, "None")
        modified_label_info.label_groups[0].insert(0, "getitune_empty_lbl")

        return super()._export_parameters.wrap(
            model_type="MaskRCNN",
            task_type="instance_segmentation",
            confidence_threshold=self.hparams.get("best_confidence_threshold", 0.05),
            iou_threshold=0.5,
            nms_execute=not (self._nms_always_embedded or self.export_nms),
            tile_config=self.tile_config if self.tile_config.enable_tiler else None,
            label_info=modified_label_info,
        )

    def test_step(self, batch: SampleBatch, batch_idx: int) -> PredictionBatch:
        """Perform a single test step on a batch of data from the test set.

        Processes the batch through the model, applies threshold filtering to predictions,
        and updates test metrics accordingly.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.
            batch_idx: The index of the current batch.

        Returns:
            PredictionBatch: The filtered prediction results for the batch.

        Raises:
            TypeError: If predictions are of type BatchLoss or if metric inputs
                    have an unsupported format.

        Note:
            The method follows a two-step process:
            1. Filters model outputs by confidence threshold
            2. Updates metrics with the filtered predictions
            Supports both single dictionary and list of dictionaries for metric inputs.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, BatchLoss):
            raise TypeError(preds)

        if isinstance(preds, torch.Tensor):
            msg = "Expected PredictionBatch, got Tensor"
            raise TypeError(msg)

        # 1. Convert predictions to metric input format
        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        # 2. Update metric
        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return preds

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return preds

        raise TypeError(metric_inputs)

    def predict_step(
        self,
        batch: SampleBatch | TileBatchData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> PredictionBatch:
        """Step function called during PyTorch Lightning Trainer's predict."""
        if self.explain_mode:
            return self._filter_outputs_by_threshold(self.forward_explain(inputs=batch))  # type: ignore[arg-type]

        outputs = self._filter_outputs_by_threshold(self.forward(inputs=batch))  # type: ignore[arg-type]

        if isinstance(outputs, BatchLoss):
            raise TypeError(outputs)

        return outputs

    @property
    def best_confidence_threshold(self) -> float:
        """Best confidence threshold to filter outputs.

        Always returns the current value from hparams, with 0.5 as fallback.
        This ensures the threshold is always up-to-date after validation updates it.
        """
        threshold = self.hparams.get("best_confidence_threshold", None)
        if threshold is None:
            # Only log warning once to avoid spam
            if not getattr(self, "_threshold_warning_logged", False):
                log.warning("There is no predefined best_confidence_threshold, 0.5 will be used as default.")
                self._threshold_warning_logged = True
            return 0.5
        return float(threshold)

    def on_load_checkpoint(self, ckpt: dict[str, Any]) -> None:
        """Load state_dict from checkpoint.

        For instance segmentation, it is needed to update confidence threshold and F1 score information when
        the metric is FMeasure.
        """
        hyper_parameters = ckpt.get("hyper_parameters", {})

        # Load best confidence threshold (legacy and new format)
        if best_confidence_threshold := ckpt.get("confidence_threshold") or hyper_parameters.get(
            "best_confidence_threshold",
            None,
        ):
            self.hparams["best_confidence_threshold"] = best_confidence_threshold
        super().on_load_checkpoint(ckpt)

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        """This function is called every epoch.

        Args:
            meter: Metric object
            key: "val" or "test"
            compute_kwargs: Additional keyword arguments for the metric computation

        """
        if key == "val":
            super()._log_metrics(meter, key)

            fmeasure = None
            if isinstance(meter, MetricCollection) and (fmeasure := getattr(meter, "FMeasure", None)):
                pass  # fmeasure is set
            elif isinstance(meter, FMeasure):
                fmeasure = meter

            if fmeasure is not None and hasattr(fmeasure, "best_confidence_threshold"):
                self.hparams["best_confidence_threshold"] = fmeasure.best_confidence_threshold

        if key == "test":
            # NOTE: Test metric logging should use `best_confidence_threshold`.
            compute_kwargs = {"best_confidence_threshold": self.best_confidence_threshold}

            super()._log_metrics(meter, key, **compute_kwargs)

    def _filter_outputs_by_threshold(self, outputs: PredictionBatch) -> PredictionBatch:
        scores = []
        bboxes = []
        labels = []
        masks = []

        for i in range(len(outputs.imgs_info)):  # type: ignore[arg-type]
            _scores = outputs.scores[i] if outputs.scores is not None else None
            _bboxes = outputs.bboxes[i] if outputs.bboxes is not None else None
            _masks = outputs.masks[i] if outputs.masks is not None else None
            _labels = outputs.labels[i] if outputs.labels is not None else None

            filtered_idx = torch.where(_scores > self.best_confidence_threshold)
            scores.append(_scores[filtered_idx])
            bboxes.append(_bboxes[filtered_idx])
            labels.append(_labels[filtered_idx])

            if _masks is not None:
                # Ensure filtered_idx is on the same device as masks
                mask_filtered_idx = tuple(idx.to(_masks.device) for idx in filtered_idx)
                masks.append(_masks[mask_filtered_idx])

        outputs.scores = scores
        outputs.bboxes = bboxes
        outputs.labels = labels
        outputs.masks = masks
        return outputs

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: PredictionBatch,  # type: ignore[override]
        inputs: SampleBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Convert the prediction entity to the format that the metric can compute and cache the ground truth.

        This function will convert mask to RLE format and cache the ground truth for the current batch.

        Args:
            preds (TorchPredBatch): Current batch predictions.
            inputs (TorchDataBatch): Current batch ground-truth inputs.

        Returns:
            dict[str, list[dict[str, Tensor]]]: The converted predictions and ground truth.
        """
        pred_info = []
        target_info = []
        for i in range(len(preds.imgs_info)):  # type: ignore[arg-type]
            bboxes = preds.bboxes[i] if preds.bboxes is not None else None
            masks = preds.masks[i] if preds.masks is not None else None
            scores = preds.scores[i] if preds.scores is not None else None
            labels = preds.labels[i] if preds.labels is not None else None

            pred_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": [encode_rle(mask) for mask in masks.data],
                    "scores": scores,
                    "labels": labels,
                },
            )
        for i in range(len(inputs.imgs_info)):  # type: ignore[arg-type]
            inputs.imgs_info[i] if inputs.imgs_info is not None else None
            bboxes = inputs.bboxes[i] if inputs.bboxes is not None else None
            masks = inputs.masks[i] if inputs.masks is not None else None
            labels = inputs.labels[i] if inputs.labels is not None else None

            if masks is None:
                msg = "Masks are required for metric computation"
                raise ValueError(msg)
            rles = [encode_rle(mask) for mask in masks.data]
            target_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": rles,
                    "labels": labels,
                },
            )
        return {"preds": pred_info, "target": target_info}

    def get_dummy_input(self, batch_size: int = 1) -> SampleBatch:  # type: ignore[override]
        """Returns a dummy input for instance segmentation model."""
        images = torch.stack([torch.rand(3, *self.data_input_params.input_size) for _ in range(batch_size)])
        img_shape = (images.shape[2], images.shape[3])
        infos = [ImageInfo(img_idx=i, img_shape=img_shape, ori_shape=img_shape) for i in range(batch_size)]
        return SampleBatch(images=images, imgs_info=infos)

    def forward_explain(self, inputs: SampleBatch) -> PredictionBatch:
        """Model forward function."""
        if isinstance(inputs, TileBatchData):
            return self.forward_tiles(inputs)

        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()

        outputs = (
            self._forward_explain_inst_seg(self.model, **self._customize_inputs(inputs))
            if self._customize_inputs != LightningInstanceSegModel._customize_inputs
            else self._forward_explain_inst_seg(self.model, inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != LightningInstanceSegModel._customize_outputs
            else outputs["predictions"]
        )

    @staticmethod
    @torch.no_grad()
    def _forward_explain_inst_seg(
        self: TwoStageDetector,  # noqa: PLW0211
        entity: SampleBatch,
        mode: str = "tensor",  # noqa: ARG004
    ) -> dict[str, Tensor]:
        """Forward func of the BaseDetector instance in ExplainableLightningInstanceSegModel().model."""
        _images = cast("torch.Tensor", entity.images)
        x = self.backbone(_images) if isinstance(self, MaskRCNN) else self.extract_feat(_images)

        feature_vector = self.feature_vector_fn(x)
        predictions = self.get_results_from_head(x, entity)

        if isinstance(predictions, tuple) and isinstance(predictions[0], Tensor):
            # Export case, consists of tensors
            # For OV task saliency map are generated on MAPI side
            saliency_map = torch.empty(1, dtype=torch.uint8)
        elif isinstance(predictions, list) and isinstance(predictions[0], (InstanceData, dict)):
            # Predict case, consists of InstanceData or dict
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
        entity: SampleBatch,
    ) -> tuple[Tensor, Tensor, Tensor] | list[InstanceData] | list[dict[str, Tensor]]:
        """Get the results from the head of the instance segmentation model.

        Args:
            x (tuple[Tensor]): The features from backbone and neck.
            data_samples (OptSampleList | None): A list of data samples.

        Returns:
            tuple[Tensor, Tensor, Tensor] | list[InstanceData]: The predicted results from the head of the model.
            Tuple for the Export case, list for the Predict case.
        """
        from getitune.backend.lightning.models.instance_segmentation.maskrcnn_tv import MaskRCNNTV
        from getitune.backend.lightning.models.instance_segmentation.rtmdet_inst import RTMDetInst

        if isinstance(self, MaskRCNNTV):
            ori_shapes = [img_info.ori_shape for img_info in entity.imgs_info]  # type: ignore[union-attr]
            img_shapes = [img_info.img_shape for img_info in entity.imgs_info]  # type: ignore[union-attr]
            image_list = ImageList(cast("torch.Tensor", entity.images), img_shapes)
            proposals, _ = self.model.rpn(image_list, x)
            detections, _ = self.model.roi_heads(
                x,
                proposals,
                image_list.image_sizes,
            )
            scale_factors = [
                img_meta.scale_factor if img_meta.scale_factor else (1.0, 1.0)  # type: ignore[union-attr]
                for img_meta in entity.imgs_info  # type: ignore[union-attr]
            ]
            return self.model.postprocess(detections, ori_shapes, scale_factors)

        if isinstance(self, RTMDetInst):
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
        # Patch method
        self.model.forward = func_type(forward_with_explain, self.model)

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
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return DataInputParams(input_size=(1024, 1024), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    @property
    def task(self) -> TaskType:
        """Return task type."""
        return TaskType.INSTANCE_SEGMENTATION
