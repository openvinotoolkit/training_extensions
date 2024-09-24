# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for 3d object detection model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision.ops import box_convert

from otx.algo.utils.mmengine_utils import load_checkpoint
from otx.core.data.dataset.kitti_3d.kitti_utils import class2angle
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DBatchPredEntity
from otx.core.metrics import MetricInput
from otx.core.metrics.ap_3d import KittiMetric
from otx.core.model.base import OTXModel
from otx.core.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable


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
        score_threshold: float = 0.2,
    ) -> None:
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
            model_type="ssd",
            task_type="detection",
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: Det3DBatchPredEntity,
        inputs: Det3DBatchDataEntity,
    ) -> MetricInput:
        """Converts the prediction entity to the format required for computing metrics."""
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
        result_list = self._decode_detections_for_kitti_format(
            detections,
            img_sizes,
            calib_matrix,
            class_names=self.label_info.label_names,
            threshold=self.score_threshold,
        )

        return {
            "preds": result_list,
            "target": inputs.kitti_label_object,  # TODO (Kirill): change it later to pre-process gt annotations here
        }

    @staticmethod
    def _decode_detections_for_kitti_format(
        self,
        dets: np.ndarray,
        img_size: np.ndarray,
        calib_matrix: list[np.ndarray],
        class_names: list[str],
        threshold: float = 0.2,
    ) -> list[dict[str, np.ndarray]]:
        """Decode the detection results for KITTI format."""

        def get_heading_angle(heading):
            heading_bin, heading_res = heading[0:12], heading[12:24]
            cls = np.argmax(heading_bin)
            res = heading_res[cls]
            return class2angle(cls, res, to_label_format=True)

        def alpha2ry(calib_matrix, alpha, u):
            """Get rotation_y by alpha + theta - 180
            alpha : Observation angle of object, ranging [-pi..pi]
            x : Object center x to the camera center (x-W/2), in pixels
            rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
            """
            cu = calib_matrix[0, 2]
            fu = calib_matrix[0, 0]

            ry = alpha + np.arctan2(u - cu, fu)

            if ry > np.pi:
                ry -= 2 * np.pi
            if ry < -np.pi:
                ry += 2 * np.pi

            return ry

        def img_to_rect(calib_matrix, u, v, depth_rect):
            """:param u: (N)
            :param v: (N)
            :param depth_rect: (N)
            :return:
            """
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
                location = img_to_rect(calib_matrix[i], x3d, y3d, depth).reshape(-1)
                location[1] += dimension[0] / 2

                # heading angle decoding
                alpha = dets[i, j, 7:31]
                alpha = get_heading_angle(dets[i, j, 7:31])
                ry = alpha2ry(calib_matrix[i], alpha, x)

                score = score * dets[i, j, -1]

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
        """Returns a dummy input for detection model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        images = [torch.rand(3, *self.input_size) for _ in range(batch_size)]
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
            boxes=[],
            labels=[],
            calib_matrix=calib_matrix,
            boxes_3d=[],
            size_2d=[],
            size_3d=[],
            depth=[],
            heading_angle=[],
            kitti_label_object=[],
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

    def forward_for_tracing(self, image: torch.Tensor, calib_matrix: torch.Tensor) -> dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(inputs=image, mode="tensor")
