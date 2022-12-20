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

import numpy as np

try:
    from openvino.model_zoo.model_api.models.instance_segmentation import MaskRCNNModel
    from openvino.model_zoo.model_api.models.ssd import SSD
except ImportError as e:
    import warnings

    warnings.warn(f"{e}: ModelAPI was not found.")


class OTXMaskRCNNModel(MaskRCNNModel):
    """OpenVINO model wrapper for OTX MaskRCNN model."""

    __model__ = "OTX_MaskRCNN"

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
        resize_mask = meta.get('resize_mask', True)

        boxes = (
            outputs[self.output_blob_name["boxes"]]
            if self.is_segmentoly
            else outputs[self.output_blob_name["boxes"]][:, :4]
        )
        scores = (
            outputs[self.output_blob_name["scores"]]
            if self.is_segmentoly
            else outputs[self.output_blob_name["boxes"]][:, 4]
        )
        masks = outputs[self.output_blob_name["masks"]]
        if self.is_segmentoly:
            classes = outputs[self.output_blob_name["labels"]].astype(np.uint32)
        else:
            classes = outputs[self.output_blob_name["labels"]].astype(np.uint32) + 1

        # Filter out detections with low confidence.
        detections_filter = scores > self.confidence_threshold  # pylint: disable=no-member
        scores = scores[detections_filter]
        boxes = boxes[detections_filter]
        masks = masks[detections_filter]
        classes = classes[detections_filter]

        scale_x = meta["resized_shape"][1] / meta["original_shape"][1]
        scale_y = meta["resized_shape"][0] / meta["original_shape"][0]
        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y

        resized_masks = []
        for box, cls, raw_mask in zip(boxes, classes, masks):
            raw_cls_mask = raw_mask[cls, ...] if self.is_segmentoly else raw_mask
            if resize_mask:
                resized_masks.append(self._segm_postprocess(box, raw_cls_mask, *meta["original_shape"][:-1]))
            else:
                resized_masks.append(raw_cls_mask)

        return scores, classes, boxes, resized_masks

    def segm_postprocess(self, *args, **kwargs):
        """Post-process for segmentation masks."""
        return self._segm_postprocess(*args, **kwargs)


class OTXSSDModel(SSD):
    """OpenVINO model wrapper for OTX SSD model."""

    __model__ = "OTX_SSD"

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
