# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import logging as log
import types
from abc import abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal
from torchvision.ops import box_convert

import torch
from model_api.tilers import DetectionTiler
from torchmetrics import Metric, MetricCollection
from torchvision import tv_tensors

from otx.algo.utils.mmengine_utils import InstanceData, load_checkpoint
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DBatchPredEntity
from otx.core.data.dataset.kitti_utils import class2angle
import numpy as np
from otx.core.data.entity.tile import OTXTileBatchDataEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.metrics import MetricCallable, MetricInput
from otx.core.metrics.fmeasure import FMeasure, MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.tile_merge import DetectionTileMerge

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import DetectionResult
    from torch import nn

    from otx.algo.detection.detectors import SingleStageDetector


class OTX3DDetectionModel(OTXModel[Det3DBatchDataEntity, Det3DBatchPredEntity]):
    """Base class for the 3d detection models used in OTX."""

    input_size: tuple[int, int]

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        self.model_name = model_name
        super().__init__(*args, **kwargs)

    def _create_model(self) -> nn.Module:
        detector = self._build_model(num_classes=self.label_info.num_classes)
        if hasattr(detector, "init_weights"):
            detector.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _customize_inputs(
        self,
        entity: Det3DBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        pass

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,
        inputs: Det3DBatchDataEntity,
    ) -> Det3DBatchPredEntity | OTXBatchLossEntity:
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
                    msg = f"Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores = []
        bboxes = []
        labels = []
        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
            if not isinstance(prediction, InstanceData):
                raise TypeError(prediction)

            scores.append(prediction.scores)  # type: ignore[attr-defined]
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,  # type: ignore[attr-defined]
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
            labels.append(prediction.labels)  # type: ignore[attr-defined]

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

            return Det3DBatchPredEntity(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=saliency_map,
                feature_vector=feature_vector,
            )

        return Det3DBatchPredEntity(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="ssd",
            task_type="detection",
            confidence_threshold=self.hparams.get("best_confidence_threshold", None),
            iou_threshold=0.5,
            tile_config=self.tile_config if self.tile_config.enable_tiler else None,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: Det3DBatchPredEntity,
        inputs: Det3DBatchDataEntity,
    ) -> MetricInput:

        depth, sigma = preds.depth[:,:,0:1], preds.depth[:,:,1:2]
        boxes = preds.boxes_3d
        xywh_2d = box_convert(preds.boxes, "xyxy", "cxcywh")

        xs3d = boxes[:, :, 0: 1]
        ys3d = boxes[:, :, 1: 2]
        xs2d = xywh_2d[:, :, 0: 1]
        ys2d = xywh_2d[:, :, 1: 2]

        batch = len(inputs.imgs_info)
        labels = preds.labels.view(batch, -1, 1)
        scores = preds.scores.view(batch, -1, 1)
        xs2d = xs2d.view(batch, -1, 1)
        ys2d = ys2d.view(batch, -1, 1)
        xs3d = xs3d.view(batch, -1, 1)
        ys3d = ys3d.view(batch, -1, 1)

        detections = torch.cat([labels, scores, xs2d, ys2d, preds.size_2d, depth, preds.heading_res, preds.size_3d, xs3d, ys3d, sigma], dim=2).detach().cpu().numpy()
        img_sizes = np.array([img_info.ori_shape for img_info in inputs.imgs_info])
        result_list = self._decode_detections_for_kitti_format(detections, img_sizes, inputs.calib, class_names=self.label_info.label_names, threshold=0.0)

        return {
            "preds": result_list,
            "target": inputs.kitti_label_object, # TODO (KIRILL): change it later to pre process metrics here
        }

    @staticmethod
    def _decode_detections_for_kitti_format(dets, img_size, calibs, class_names, threshold=0.2):
        '''
        input: dets, numpy array, shape in [batch x max_dets x dim]
        input: img_info, dict, necessary information of input images
        input: calibs, corresponding calibs for the input batch
        output:
        '''
        def get_heading_angle(heading):
            heading_bin, heading_res = heading[0:12], heading[12:24]
            cls = np.argmax(heading_bin)
            res = heading_res[cls]
            return class2angle(cls, res, to_label_format=True)

        results = []
        for i in range(dets.shape[0]):  # batch
            preds = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score' : [],
            }
            for j in range(dets.shape[1]):  # max_dets
                cls_id = int(dets[i, j, 0])
                score = dets[i, j, 1]
                if score < threshold:
                    continue
                # 2d bboxs decoding
                x = dets[i, j, 2] * img_size[i][0]
                y = dets[i, j, 3] * img_size[i][1]
                w = dets[i, j, 4] * img_size[i][0]
                h = dets[i, j, 5] * img_size[i][1]
                bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

                # 3d bboxs decoding
                # depth decoding
                depth = dets[i, j, 6]

                # dimensions decoding
                dimensions = dets[i, j, 31:34]

                # positions decoding
                x3d = dets[i, j, 34] * img_size[i][0]
                y3d = dets[i, j, 35] * img_size[i][1]
                locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
                locations[1] += dimensions[0] / 2

                # heading angle decoding
                alpha = dets[i, j, 7:31]
                alpha = get_heading_angle(dets[i, j, 7:31])
                ry = calibs[i].alpha2ry(alpha, x)

                score = dets[i, j, 1] * dets[i, j, -1]

                preds["name"].append(class_names[cls_id])
                preds["alpha"].append(alpha)
                preds["bbox"].append(bbox)
                preds["dimensions"].append(dimensions.tolist())
                preds["location"].append(locations.tolist())
                preds["rotation_y"].append(ry)
                preds["score"].append(score) # can be discarded I think

            for key, value in preds.items():
                preds[key] = np.array(value)

            results.append(preds)

        return results

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

    @property
    def best_confidence_threshold(self) -> float:
        """Best confidence threshold to filter outputs."""
        if not hasattr(self, "_best_confidence_threshold"):
            self._best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
            if self._best_confidence_threshold is None:
                log.warning("There is no predefined best_confidence_threshold, 0.5 will be used as default.")
                self._best_confidence_threshold = 0.5
        return self._best_confidence_threshold

    def get_dummy_input(self, batch_size: int = 1) -> Det3DBatchDataEntity:
        """Returns a dummy input for detection model."""
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
        return Det3DBatchDataEntity(batch_size, images, infos, bboxes=[], labels=[])

    def get_classification_layers(self, prefix: str = "model.") -> dict[str, dict[str, int]]:
        """Get final classification layer information for incremental learning case."""
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

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(inputs=image, mode="tensor")
