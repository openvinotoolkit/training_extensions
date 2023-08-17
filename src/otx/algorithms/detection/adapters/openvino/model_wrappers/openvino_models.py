"""OTXMaskRCNNModel & OTXSSDModel of OTX Detection."""

# Copyright (C) 2022 Intel Corporation
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

from typing import Dict

import cv2
import numpy as np
from openvino.model_api.models.instance_segmentation import MaskRCNNModel, _expand_box, _segm_postprocess
from openvino.model_api.models.ssd import SSD, find_layer_by_name
from openvino.model_api.models.utils import Detection


class OTXMaskRCNNModel(MaskRCNNModel):
    """OpenVINO model wrapper for OTX MaskRCNN model."""

    __model__ = "OTX_MaskRCNN"

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self.resize_mask = True

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        """Checks whether the number of model inputs/outputs is supported.

        Args:
            number_of_inputs (int, Tuple(int)): number of inputs supported by wrapper.
              Use -1 to omit the check
            number_of_outputs (int, Tuple(int)): number of outputs supported by wrapper.
              Use -1 to omit the check

        Raises:
            WrapperError: if the model has unsupported number of inputs/outputs
        """
        super()._check_io_number(number_of_inputs, -1)

    def _get_outputs(self):
        output_match_dict = {}
        output_names = ["boxes", "labels", "masks", "feature_vector", "saliency_map"]
        for output_name in output_names:
            for node_name, node_meta in self.outputs.items():
                if output_name in node_meta.names:
                    output_match_dict[output_name] = node_name
                    break
        return output_match_dict

    def postprocess(self, outputs, meta):
        """Post process function for OTX MaskRCNN model."""

        # pylint: disable-msg=too-many-locals
        # FIXME: here, batch dim of IR must be 1
        boxes = outputs[self.output_blob_name["boxes"]]
        if boxes.shape[0] == 1:
            boxes = boxes.squeeze(0)
        assert boxes.ndim == 2
        masks = outputs[self.output_blob_name["masks"]]
        if masks.shape[0] == 1:
            masks = masks.squeeze(0)
        assert masks.ndim == 3
        classes = outputs[self.output_blob_name["labels"]].astype(np.uint32)
        if classes.shape[0] == 1:
            classes = classes.squeeze(0)
        assert classes.ndim == 1
        if self.is_segmentoly:
            scores = outputs[self.output_blob_name["scores"]]
        else:
            scores = boxes[:, 4]
            boxes = boxes[:, :4]
            classes += 1

        # Filter out detections with low confidence.
        detections_filter = scores > self.confidence_threshold  # pylint: disable=no-member
        scores = scores[detections_filter]
        boxes = boxes[detections_filter]
        masks = masks[detections_filter]
        classes = classes[detections_filter]

        inputImgWidth, inputImgHeight = (
            meta["original_shape"][1],
            meta["original_shape"][0],
        )
        invertedScaleX, invertedScaleY = (
            inputImgWidth / self.orig_width,
            inputImgHeight / self.orig_height,
        )
        padLeft, padTop = 0, 0
        if "fit_to_window" == self.resize_type or "fit_to_window_letterbox" == self.resize_type:
            invertedScaleX = invertedScaleY = max(invertedScaleX, invertedScaleY)
            if "fit_to_window_letterbox" == self.resize_type:
                padLeft = (self.orig_width - round(inputImgWidth / invertedScaleX)) // 2
                padTop = (self.orig_height - round(inputImgHeight / invertedScaleY)) // 2

        boxes -= (padLeft, padTop, padLeft, padTop)
        boxes *= (invertedScaleX, invertedScaleY, invertedScaleX, invertedScaleY)
        np.around(boxes, out=boxes)
        np.clip(
            boxes,
            0.0,
            [inputImgWidth, inputImgHeight, inputImgWidth, inputImgHeight],
            out=boxes,
        )

        resized_masks = []
        for box, cls, raw_mask in zip(boxes, classes, masks):
            raw_cls_mask = raw_mask[cls, ...] if self.is_segmentoly else raw_mask
            if self.resize_mask:
                resized_masks.append(_segm_postprocess(box, raw_cls_mask, *meta["original_shape"][:-1]))
            else:
                resized_masks.append(raw_cls_mask)

        return scores, classes, boxes, resized_masks

    def get_saliency_map_from_prediction(self, outputs, meta, num_classes):
        """Post process function for saliency map of OTX MaskRCNN model."""
        boxes = outputs[self.output_blob_name["boxes"]]
        if boxes.shape[0] == 1:
            boxes = boxes.squeeze(0)
        scores = boxes[:, 4]
        boxes = boxes[:, :4]
        masks = outputs[self.output_blob_name["masks"]]
        if masks.shape[0] == 1:
            masks = masks.squeeze(0)
        classes = outputs[self.output_blob_name["labels"]].astype(np.uint32)
        if classes.shape[0] == 1:
            classes = classes.squeeze(0)

        scale_x = meta["resized_shape"][0] / meta["original_shape"][1]
        scale_y = meta["resized_shape"][1] / meta["original_shape"][0]
        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y

        saliency_maps = [None for _ in range(num_classes)]
        for box, score, cls, raw_mask in zip(boxes, scores, classes, masks):
            resized_mask = self._resize_mask(box, raw_mask * score, *meta["original_shape"][:-1])
            if saliency_maps[cls] is None:
                saliency_maps[cls] = [resized_mask]
            else:
                saliency_maps[cls].append(resized_mask)

        saliency_maps = self._average_and_normalize(saliency_maps, num_classes)
        return saliency_maps

    def _resize_mask(self, box, raw_cls_mask, im_h, im_w):
        # Add zero border to prevent upsampling artifacts on segment borders.
        raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), "constant", constant_values=0)
        extended_box = _expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask.astype(np.float32), (w, h))
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.float32)
        im_mask[y0:y1, x0:x1] = raw_cls_mask[
            (y0 - extended_box[1]) : (y1 - extended_box[1]), (x0 - extended_box[0]) : (x1 - extended_box[0])
        ]
        return im_mask

    @staticmethod
    def _average_and_normalize(saliency_maps, num_classes):
        for i in range(num_classes):
            if saliency_maps[i] is not None:
                saliency_maps[i] = np.array(saliency_maps[i]).mean(0)

        for i in range(num_classes):
            per_class_map = saliency_maps[i]
            if per_class_map is not None:
                max_values = np.max(per_class_map)
                per_class_map = 255 * (per_class_map) / (max_values + 1e-12)
                per_class_map = per_class_map.astype(np.uint8)
                saliency_maps[i] = per_class_map
        return saliency_maps

    def segm_postprocess(self, *args, **kwargs):
        """Post-process for segmentation masks."""
        return _segm_postprocess(*args, **kwargs)

    def disable_mask_resizing(self):
        """Disable mask resizing.

        There is no need to resize mask in tile as it will be processed at the end.
        """
        self.resize_mask = False


class OTXSSDModel(SSD):
    """OpenVINO model wrapper for OTX SSD model."""

    __model__ = "OTX_SSD"

    def __init__(self, model_adapter, configuration=None, preload=False):
        # pylint: disable-next=bad-super-call
        super(SSD, self).__init__(model_adapter, configuration, preload)
        self.image_info_blob_name = self.image_info_blob_names[0] if len(self.image_info_blob_names) == 1 else None
        self.output_parser = BatchBoxesLabelsParser(
            self.outputs,
            self.inputs[self.image_blob_name].shape[2:][::-1],
        )

    def _get_outputs(self) -> Dict:
        """Match the output names with graph node index."""
        output_match_dict = {}
        output_names = ["boxes", "labels", "feature_vector", "saliency_map"]
        for output_name in output_names:
            for node_name, node_meta in self.outputs.items():
                if output_name in node_meta.names:
                    output_match_dict[output_name] = node_name
                    break
        return output_match_dict


class BatchBoxesLabelsParser:
    """Batched output parser."""

    def __init__(self, layers, input_size, labels_layer="labels", default_label=0):
        try:
            self.labels_layer = find_layer_by_name(labels_layer, layers)
        except ValueError:
            self.labels_layer = None
            self.default_label = default_label

        try:
            self.bboxes_layer = self.find_layer_bboxes_output(layers)
        except ValueError:
            self.bboxes_layer = find_layer_by_name("boxes", layers)

        self.input_size = input_size

    @staticmethod
    def find_layer_bboxes_output(layers):
        """find_layer_bboxes_output."""
        filter_outputs = [name for name, data in layers.items() if len(data.shape) == 3 and data.shape[-1] == 5]
        if not filter_outputs:
            raise ValueError("Suitable output with bounding boxes is not found")
        if len(filter_outputs) > 1:
            raise ValueError("More than 1 candidate for output with bounding boxes.")
        return filter_outputs[0]

    def __call__(self, outputs):
        """Parse bboxes."""
        # FIXME: here, batch dim of IR must be 1
        bboxes = outputs[self.bboxes_layer]
        if bboxes.shape[0] == 1:
            bboxes = bboxes.squeeze(0)
        assert bboxes.ndim == 2
        scores = bboxes[:, 4]
        bboxes = bboxes[:, :4]
        bboxes[:, 0::2] /= self.input_size[0]
        bboxes[:, 1::2] /= self.input_size[1]
        if self.labels_layer:
            labels = outputs[self.labels_layer]
        else:
            labels = np.full(len(bboxes), self.default_label, dtype=bboxes.dtype)
        if labels.shape[0] == 1:
            labels = labels.squeeze(0)

        detections = [Detection(*bbox, score, label) for label, score, bbox in zip(labels, scores, bboxes)]
        return detections
