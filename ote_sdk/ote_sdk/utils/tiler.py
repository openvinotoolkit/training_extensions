"""
Tiling Module
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from itertools import product
from typing import List, Tuple

import numpy as np
from openvino.model_zoo.model_api.models import Model

from ote_sdk.utils.detection_utils import detection2array
from ote_sdk.utils.nms import multiclass_nms


class Tiler:
    """Tile Image into (non)overlapping Patches. Images are tiled in order to efficiently process large images.

    Args:
        tile_size: Tile dimension for each patch
        overlap: Overlap between adjacent tile
    """

    def __init__(
        self, tile_size: int, overlap: float, max_number: int, model: Model
    ) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.max_number = max_number
        self.model = model

    def tile(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Tiles an input image to either overlapping, non-overlapping or random patches.

        Args:
            image: Input image to tile.

        Returns:
            Tiles coordinates
        """
        height, width = image.shape[:2]

        coords = [(0, 0, width, height)]
        for (loc_j, loc_i) in product(
            range(0, width - self.tile_size + 1, self.stride),
            range(0, height - self.tile_size + 1, self.stride),
        ):
            coords.append(
                (loc_j, loc_i, loc_j + self.tile_size, loc_i + self.tile_size)
            )
        return coords

    def predict(self, image: np.ndarray, segm: bool = False):
        """Predict by cropping full image to tiles

        Args:
            image (np.ndarray): full size image
            segm (bool, optional): return mask if enabled. Defaults to False.

        Returns:
            detection: prediction results
            features: saliency map and feature vector
        """
        detections = np.empty((0, 6), dtype=np.float32)
        features = [None, None]
        masks = []
        for i, coord in enumerate(self.tile(image)):
            x1, y1, x2, y2 = coord
            tile_dict, tile_meta = self.model.preprocess(image[y1:y2, x1:x2])
            if segm:
                tile_meta["resize_mask"] = False
            raw_predictions = self.model.infer_sync(tile_dict)
            output = self.model.postprocess(raw_predictions, tile_meta)
            # cache full image feature vector and saliency map at 0 index
            if i == 0:
                if (
                    "feature_vector" in raw_predictions
                    or "saliency_map" in raw_predictions
                ):
                    features = [
                        raw_predictions["feature_vector"].reshape(-1),
                        raw_predictions["saliency_map"],
                    ]
            if segm:
                tile_scores, tile_labels, tile_boxes, tile_masks = output
                tile_boxes += np.tile([x1, y1], 2)
                output = np.concatenate(
                    (
                        tile_labels[:, np.newaxis],
                        tile_scores[:, np.newaxis],
                        tile_boxes,
                    ),
                    -1,
                )
                masks.extend(tile_masks)
            else:
                output = detection2array(output)
                output[:, 2:] += np.tile([x1, y1], 2)
            detections = np.append(detections, output, axis=0)

        if np.prod(detections.shape):
            labels = detections[:, 0]
            scores = detections[:, 1]
            boxes = detections[:, 2:]
            keep = multiclass_nms(scores, labels, boxes, max_num=self.max_number)

            detections = detections[keep]
            labels = labels[keep]
            scores = scores[keep]
            boxes = boxes[keep]

            if segm:
                masks = [masks[keep_idx] for keep_idx in keep]
                for i, (box, mask) in enumerate(zip(boxes, masks)):
                    masks[i] = self.model._segm_postprocess(
                        box, mask, *image.shape[:-1]
                    )
                detections = labels, scores, boxes, masks
        return detections, features
