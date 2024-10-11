# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for 3d object detection model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from model_api.models import ImageModel
from torchvision.ops import box_convert

from otx.algo.object_detection_3d.utils.utils import box_cxcylrtb_to_xyxy
from otx.algo.utils.mmengine_utils import load_checkpoint
from otx.core.data.dataset.utils.kitti_utils import class2angle
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DBatchPredEntity
from otx.core.metrics import MetricInput
from otx.core.metrics.average_precision_3d import KittiMetric
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.adapters.inference_adapter import InferenceAdapter
    from torch import nn

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class OTX3DDetectionModel(OTXModel[Det3DBatchDataEntity, Det3DBatchPredEntity]):
    """Base class for the 3d detection models used in OTX."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    load_from: str | None

    def __init__(
        self,
        label_info: LabelInfoTypes,
        model_name: str,
        input_size: tuple[int, int],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = KittiMetric,
        torch_compile: bool = False,
        score_threshold: float = 0.1,
    ) -> None:
        """Initialize the 3d detection model."""
        self.model_name = model_name
        self.score_threshold = score_threshold
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        """Creates the model."""
        detector = self._build_model(num_classes=self.label_info.num_classes)
        if hasattr(detector, "init_weights"):
            detector.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="mono_3d_det",
            task_type="3d_detection",
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: Det3DBatchPredEntity,
        inputs: Det3DBatchDataEntity,
    ) -> MetricInput:
        return _convert_pred_entity_to_compute_metric(preds, inputs, self.label_info.label_names, self.score_threshold)

    @staticmethod
    def decode_detections_for_kitti_format(
        dets: np.ndarray,
        img_size: np.ndarray,
        calib_matrix: list[np.ndarray],
        class_names: list[str],
        threshold: float = 0.2,
    ) -> list[dict[str, np.ndarray]]:
        """Decode the detection results for KITTI format."""

        def _get_heading_angle(heading: np.ndarray) -> np.ndarray:
            """Get heading angle from the prediction."""
            heading_bin, heading_res = heading[0:12], heading[12:24]
            cls = np.argmax(heading_bin)
            res = heading_res[cls]
            return class2angle(cls, res, to_label_format=True)

        def _alpha2ry(calib_matrix: np.ndarray, alpha: np.ndarray, u: np.ndarray) -> np.ndarray:
            """Get rotation_y by alpha + theta - 180."""
            cu = calib_matrix[0, 2]
            fu = calib_matrix[0, 0]

            ry = alpha + np.arctan2(u - cu, fu)

            if ry > np.pi:
                ry -= 2 * np.pi
            if ry < -np.pi:
                ry += 2 * np.pi

            return ry

        def _img_to_rect(calib_matrix: np.ndarray, u: np.ndarray, v: np.ndarray, depth_rect: np.ndarray) -> np.ndarray:
            """Transform image coordinates to the rectangle coordinates."""
            cu = calib_matrix[0, 2]
            cv = calib_matrix[1, 2]
            fu = calib_matrix[0, 0]
            fv = calib_matrix[1, 1]
            tx = calib_matrix[0, 3] / (-fu)
            ty = calib_matrix[1, 3] / (-fv)

            x = ((u - cu) * depth_rect) / fu + tx
            y = ((v - cv) * depth_rect) / fv + ty
            return np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)

        results = []
        for i in range(dets.shape[0]):  # batch
            names = []
            alphas = []
            bboxes = []
            dimensions = []
            locations = []
            rotation_y = []
            scores = []

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
                bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

                # 3d bboxs decoding
                # depth decoding
                depth = dets[i, j, 6]

                # dimensions decoding
                dimension = dets[i, j, 31:34]

                # positions decoding
                x3d = dets[i, j, 34] * img_size[i][0]
                y3d = dets[i, j, 35] * img_size[i][1]
                location = _img_to_rect(calib_matrix[i], x3d, y3d, depth).reshape(-1)
                location[1] += dimension[0] / 2

                # heading angle decoding
                alpha = dets[i, j, 7:31]
                alpha = _get_heading_angle(dets[i, j, 7:31])
                ry = _alpha2ry(calib_matrix[i], alpha, x)

                names.append(class_names[cls_id])
                alphas.append(alpha)
                bboxes.append(bbox)
                dimensions.append(np.array([dimension[2], dimension[0], dimension[1]]))
                locations.append(location)
                rotation_y.append(ry)
                scores.append(score)

            results.append(
                {
                    "name": np.array(names),
                    "alpha": np.array(alphas),
                    "bbox": np.array(bboxes).reshape(-1, 4),
                    "dimensions": np.array(dimensions).reshape(-1, 3),
                    "location": np.array(locations).reshape(-1, 3),
                    "rotation_y": np.array(rotation_y),
                    "score": np.array(scores),
                },
            )

        return results

    def get_dummy_input(self, batch_size: int = 1) -> Det3DBatchDataEntity:
        """Returns a dummy input for 3d object detection model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        images = torch.rand(batch_size, 3, *self.input_size)
        calib_matrix = [torch.rand(3, 4) for _ in range(batch_size)]
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return Det3DBatchDataEntity(
            batch_size,
            images,
            infos,
            boxes=[torch.Tensor(0)] * batch_size,
            labels=[torch.LongTensor(0)] * batch_size,
            calib_matrix=calib_matrix,
            boxes_3d=[torch.LongTensor(0)] * batch_size,
            size_2d=[],
            size_3d=[torch.LongTensor(0)] * batch_size,
            depth=[torch.LongTensor(0)] * batch_size,
            heading_angle=[torch.LongTensor(0)] * batch_size,
            original_kitti_format=[],
        )

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


class MonoDETRModel(ImageModel):
    """A wrapper for MonoDETR 3d object detection model."""

    __model__ = "mono_3d_det"

    def __init__(self, inference_adapter: InferenceAdapter, configuration: dict[str, Any], preload: bool = False):
        """Initializes a 3d detection model.

        Args:
            inference_adapter (InferenceAdapter): inference adapter containing the underlying model.
            configuration (dict, optional): configuration overrides the model parameters (see parameters() method).
              Defaults to dict().
            preload (bool, optional): forces inference adapter to load the model. Defaults to False.
        """
        super().__init__(inference_adapter, configuration, preload)
        self._check_io_number(3, 5)

    def preprocess(self, inputs: dict[str, np.ndarray]) -> tuple[dict[str, Any], ...]:
        """Preprocesses the input data for the model.

        Args:
            inputs (dict[str, np.ndarray]): a dict with image, calibration matrix, and image size

        Returns:
            tuple[dict[str, Any], ...]: a tuple with the preprocessed inputs and meta information
        """
        return {
            self.image_blob_name: inputs["image"][None],
            "calib_matrix": inputs["calib"],
            "img_sizes": inputs["img_size"][None],
        }, {
            "original_shape": inputs["image"].shape,
            "resized_shape": (self.w, self.h, self.c),
        }

    def _get_inputs(self) -> tuple[list[Any], list[Any]]:
        """Defines the model inputs for images and additional info.

        Raises:
            WrapperError: if the wrapper failed to define appropriate inputs for images

        Returns:
            - list of inputs names for images
            - list of inputs names for additional info
        """
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 4:
                image_blob_names.append(name)
            elif len(metadata.shape) == 2:
                image_info_blob_names.append(name)

        if not image_blob_names:
            self.raise_error(
                "Failed to identify the input for the image: no 4D input layer found",
            )
        return image_blob_names, image_info_blob_names

    def postprocess(
        self,
        outputs: dict[str, np.ndarray],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Applies SCC decoded to the model outputs.

        Args:
            outputs (dict[str, np.ndarray]): raw outputs of the model
            meta (dict[str, Any]): meta information about the input data

        Returns:
            dict[str, Any]: postprocessed model outputs
        """
        result = {}
        for k in outputs:
            result[k] = np.copy(outputs[k])

        return result


class OV3DDetectionModel(OVModel[Det3DBatchDataEntity, Det3DBatchPredEntity]):
    """3d detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX 3d detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "mono_3d_det",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = KittiMetric,
        score_threshold: float = 0.2,
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
        self.score_threshold = score_threshold

    def _customize_inputs(
        self,
        entity: Det3DBatchDataEntity,
    ) -> dict[str, Any]:
        img_sizes = np.array([img_info.ori_shape for img_info in entity.imgs_info])
        images = [np.transpose(im.cpu().numpy(), (1, 2, 0)) for im in entity.images]

        return {
            "images": images,
            "calibs": [p2.unsqueeze(0).cpu().numpy() for p2 in entity.calib_matrix],
            "targets": [],
            "img_sizes": img_sizes,
            "mode": "predict",
        }

    def _customize_outputs(
        self,
        outputs: list[NamedTuple],
        inputs: Det3DBatchDataEntity,
    ) -> Det3DBatchPredEntity | OTXBatchLossEntity:
        stacked_outputs: dict[str, Any] = {}

        for output in outputs:
            for k in output:
                if k in stacked_outputs:
                    stacked_outputs[k] = torch.cat((stacked_outputs[k], torch.tensor(output[k])), 0)
                else:
                    stacked_outputs[k] = torch.tensor(output[k])

        labels, scores, size_3d, heading_angle, boxes_3d, depth = self.extract_dets_from_outputs(stacked_outputs)
        # bbox 2d decoding
        boxes_2d = box_cxcylrtb_to_xyxy(boxes_3d)
        xywh_2d = box_convert(boxes_2d, "xyxy", "cxcywh")
        # size 2d decoding
        size_2d = xywh_2d[:, :, 2:4]

        return Det3DBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            calib_matrix=inputs.calib_matrix,
            boxes=boxes_2d,
            labels=labels,
            boxes_3d=boxes_3d,
            size_2d=size_2d,
            size_3d=size_3d,
            depth=depth,
            heading_angle=heading_angle,
            scores=scores,
            original_kitti_format=[None],
        )

    def _forward(self, inputs: Det3DBatchDataEntity) -> Det3DBatchPredEntity:
        """Model forward function."""
        all_inputs = self._customize_inputs(inputs)

        model_ready_inputs = []
        for image, calib, img_size in zip(all_inputs["images"], all_inputs["calibs"], all_inputs["img_sizes"]):
            model_ready_inputs.append(
                {
                    "image": image,
                    "calib": calib,
                    "img_size": img_size,
                },
            )

        if self.async_inference:
            outputs = self.model.infer_batch(model_ready_inputs)
        else:
            outputs = []
            for model_input in model_ready_inputs:
                outputs.append(self.model(model_input))

        customized_outputs = self._customize_outputs(outputs, inputs)

        if isinstance(customized_outputs, OTXBatchLossEntity):
            raise TypeError(customized_outputs)

        return customized_outputs

    def transform_fn(self, data_batch: Det3DBatchDataEntity) -> dict:
        """Data transform function for PTQ."""
        all_inputs = self._customize_inputs(data_batch)
        image = all_inputs["images"][0]
        model = self.model
        resized_image = model.resize(image, (model.w, model.h))
        resized_image = model.input_transform(resized_image)

        return {
            "images": model._change_layout(resized_image),  # noqa: SLF001,
            "calib_matrix": all_inputs["calibs"][0],
            "img_sizes": all_inputs["img_sizes"][0][None],
        }

    @staticmethod
    def extract_dets_from_outputs(outputs: dict[str, torch.Tensor], topk: int = 50) -> tuple[torch.Tensor, ...]:
        """Extract detection results from model outputs."""
        # b, q, c
        out_logits = outputs["scores"]
        out_bbox = outputs["boxes_3d"]

        prob = out_logits
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

        # final scores
        scores = topk_values
        # final indexes
        topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
        # final labels
        labels = topk_indexes % out_logits.shape[2]

        heading = outputs["heading_angle"]
        size_3d = outputs["size_3d"]
        depth = outputs["depth"]
        # decode boxes
        boxes_3d = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4
        # heading angle decoding
        heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
        # depth decoding
        depth = torch.gather(depth, 1, topk_boxes.repeat(1, 1, 2))
        # 3d dims decoding
        size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))
        # 2d boxes of the corners decoding

        return labels, scores, size_3d, heading, boxes_3d, depth

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: Det3DBatchPredEntity,
        inputs: Det3DBatchDataEntity,
    ) -> MetricInput:
        return _convert_pred_entity_to_compute_metric(preds, inputs, self.label_info.label_names, self.score_threshold)


def _convert_pred_entity_to_compute_metric(
    preds: Det3DBatchPredEntity,
    inputs: Det3DBatchDataEntity,
    label_names: list[str],
    score_threshold: float,
) -> MetricInput:
    """Converts the prediction entity to the format required for computing metrics.

    Args:
        preds (Det3DBatchPredEntity): Prediction entity.
        inputs (Det3DBatchDataEntity): Input data entity.
        label_names (list[str]): List of label names.
        score_threshold (float): Score threshold for filtering the predictions.
    """
    boxes = preds.boxes_3d
    # bbox 2d decoding
    xywh_2d = box_convert(preds.boxes, "xyxy", "cxcywh")

    xs3d = boxes[:, :, 0:1]
    ys3d = boxes[:, :, 1:2]
    xs2d = xywh_2d[:, :, 0:1]
    ys2d = xywh_2d[:, :, 1:2]

    batch = len(boxes)
    labels = preds.labels.view(batch, -1, 1)
    scores = preds.scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = (
        torch.cat(
            [
                labels,
                scores,
                xs2d,
                ys2d,
                preds.size_2d,
                preds.depth[:, :, 0:1],
                preds.heading_angle,
                preds.size_3d,
                xs3d,
                ys3d,
                torch.exp(-preds.depth[:, :, 1:2]),
            ],
            dim=2,
        )
        .detach()
        .cpu()
        .numpy()
    )

    img_sizes = np.array([img_info.ori_shape for img_info in inputs.imgs_info])
    calib_matrix = [p2.detach().cpu().numpy() for p2 in inputs.calib_matrix]
    result_list = OTX3DDetectionModel.decode_detections_for_kitti_format(
        detections,
        img_sizes,
        calib_matrix,
        class_names=label_names,
        threshold=score_threshold,
    )

    return {
        "preds": result_list,
        "target": inputs.original_kitti_format,  # type: ignore[dict-item]
    }
