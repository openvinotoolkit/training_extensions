"""OTX_MMROTATED_Model of OTX Rotated Detection."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import numpy as np
from mmrotate.core import obb2poly_np
from openvino.model_api.models.ssd import SSD, find_layer_by_name


class QuadrilateralDetection:
    """Quadrilateral detection representation."""

    def __init__(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        score: float,
        id: int,
        str_label: str = None,
    ):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.score = score
        self.id = int(id)
        self.str_label = str_label

    def __str__(self):
        """Returns string representation of the detection."""
        return (
            f"{self.x0}, {self.y0}, {self.x1}, {self.y1}, {self.x2}, {self.y2}, {self.x3}, {self.y3}, "
            "{self.id}, {self.str_label}, {self.score:.3f}"
        )


class OTX_MMROTATED_Model(SSD):
    """OpenVINO model wrapper for OTX SSD model."""

    __model__ = "OTX_MMROTATED_Model"

    def __init__(self, model_adapter, configuration=None, preload=False):
        # pylint: disable-next=bad-super-call
        super(SSD, self).__init__(model_adapter, configuration, preload)
        self.image_info_blob_name = self.image_info_blob_names[0] if len(self.image_info_blob_names) == 1 else None
        self.output_parser = RotateBoxesLabelsParser(
            self.outputs,
            self.inputs[self.image_blob_name].shape[2:][::-1],
            configuration["angle_version"],
        )

    def _resize_detections(self, detections, meta):
        """Resizes detection bounding boxes according to initial image shape.

        It implements image resizing depending on the set `resize_type`(see `ImageModel` for details).
        Next, it applies bounding boxes clipping.

        Args:
            detections (List[Detection]): list of detections with coordinates in normalized form
            meta (dict): the input metadata obtained from `preprocess` method

        Returns:
            - list of detections with resized and clipped coordinates to fit the initial image
        """
        input_img_height, input_img_width = meta["original_shape"][:2]
        inverted_scale_x = input_img_width / self.w
        inverted_scale_y = input_img_height / self.h
        pad_left = 0
        pad_top = 0
        if "fit_to_window" == self.resize_type or "fit_to_window_letterbox" == self.resize_type:
            inverted_scale_x = inverted_scale_y = max(inverted_scale_x, inverted_scale_y)
            if "fit_to_window_letterbox" == self.resize_type:
                pad_left = (self.w - round(input_img_width / inverted_scale_x)) // 2
                pad_top = (self.h - round(input_img_height / inverted_scale_y)) // 2

        for detection in detections:
            detection.x0 = min(
                max(round((detection.x0 * self.w - pad_left) * inverted_scale_x), 0),
                input_img_width,
            )
            detection.y0 = min(
                max(round((detection.y0 * self.h - pad_top) * inverted_scale_y), 0),
                input_img_height,
            )
            detection.x1 = min(
                max(round((detection.x1 * self.w - pad_left) * inverted_scale_x), 0),
                input_img_width,
            )
            detection.y1 = min(
                max(round((detection.y1 * self.h - pad_top) * inverted_scale_y), 0),
                input_img_height,
            )
            detection.x2 = min(
                max(round((detection.x2 * self.w - pad_left) * inverted_scale_x), 0),
                input_img_width,
            )
            detection.y2 = min(
                max(round((detection.y2 * self.h - pad_top) * inverted_scale_y), 0),
                input_img_height,
            )
            detection.x3 = min(
                max(round((detection.x3 * self.w - pad_left) * inverted_scale_x), 0),
                input_img_width,
            )
            detection.y3 = min(
                max(round((detection.y3 * self.h - pad_top) * inverted_scale_y), 0),
                input_img_height,
            )
        return detections


class RotateBoxesLabelsParser:
    """Parser for rotated boxes and labels."""

    def __init__(self, layers, input_size, angle_version, labels_layer="labels", default_label=0):
        try:
            self.labels_layer = find_layer_by_name(labels_layer, layers)
        except ValueError:
            self.labels_layer = None
            self.default_label = default_label

        try:
            self.bboxes_layer = self.find_layer_rboxes_output(layers)
        except ValueError:
            self.bboxes_layer = find_layer_by_name("boxes", layers)
        self.angle_version = angle_version
        self.input_size = input_size

    @staticmethod
    def find_layer_rboxes_output(layers):
        """Find output layer with format as cx, cy, w, h, angle, score."""
        filter_outputs = [
            name
            for name, data in layers.items()
            if (len(data.shape) == 2 or len(data.shape) == 3) and data.shape[-1] == 6
        ]
        if not filter_outputs:
            raise ValueError("Suitable output with bounding boxes is not found")
        if len(filter_outputs) > 1:
            raise ValueError("More than 1 candidate for output with bounding boxes.")
        return filter_outputs[0]

    def __call__(self, outputs):
        """Parse rboxes."""
        rboxes = outputs[self.bboxes_layer]
        if rboxes.shape[0] == 1:
            rboxes = rboxes.squeeze(0)
        assert rboxes.ndim == 2

        polys = obb2poly_np(rboxes, self.angle_version)
        polygons = polys[:, :-1]
        scores = polys[:, -1]
        polygons[:, 0::2] /= self.input_size[0]
        polygons[:, 1::2] /= self.input_size[1]
        if self.labels_layer:
            labels = outputs[self.labels_layer]
        else:
            labels = np.full(len(polys), self.default_label, dtype=polys.dtype)
        if labels.shape[0] == 1:
            labels = labels.squeeze(0)

        detections = [
            QuadrilateralDetection(*poly, score, label) for poly, score, label in zip(polygons, scores, labels)
        ]
        return detections
