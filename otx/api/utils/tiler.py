"""Tiling Module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
from itertools import product
from typing import Any, List, Tuple, Union

import numpy as np

from otx.api.utils.detection_utils import detection2array
from otx.api.utils.nms import multiclass_nms


class Tiler:
    """Tile Image into (non)overlapping Patches. Images are tiled in order to efficiently process large images.

    Args:
        tile_size: Tile dimension for each patch
        overlap: Overlap between adjacent tile
        max_number: max number of prediction per image
        segm: enable instance segmentation mask output
    """

    def __init__(
        self,
        tile_size: int,
        overlap: float,
        max_number: int,
        model: Any,
        segm: bool = False,
    ) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.max_number = max_number
        self.model = model
        self.segm = segm

    def tile(self, image: np.ndarray) -> List[List[int]]:
        """Tiles an input image to either overlapping, non-overlapping or random patches.

        Args:
            image: Input image to tile.

        Returns:
            Tiles coordinates
        """
        height, width = image.shape[:2]

        coords = [[0, 0, width, height]]
        for (loc_j, loc_i) in product(
            range(0, width - self.tile_size + 1, self.stride),
            range(0, height - self.tile_size + 1, self.stride),
        ):
            coords.append([loc_j, loc_i, loc_j + self.tile_size, loc_i + self.tile_size])
        return coords

    def predict(self, image: np.ndarray):
        """Predict by cropping full image to tiles.

        Args:
            image (np.ndarray): full size image

        Returns:
            detection: prediction results
            features: saliency map and feature vector
        """
        detections = np.empty((0, 6), dtype=np.float32)
        features = (None, None)
        masks: List[np.ndarray] = []
        for i, coord in enumerate(self.tile(image)):
            feats, output = self.predict_tile(image, coord, masks, i == 0)
            detections = np.append(detections, output, axis=0)
            # cache full image feature vector and saliency map at 0 index
            if i == 0:
                features = copy.deepcopy(feats)

        if np.prod(detections.shape):
            detections, keep = multiclass_nms(detections, max_num=self.max_number)
            if self.segm:
                masks = [masks[keep_idx] for keep_idx in keep]
                self.resize_masks(masks, detections, image.shape)
                detections = *Tiler.detection2tuple(detections), masks
        return detections, features

    def resize_masks(self, masks: List, dets: np.ndarray, shape: List[int]):
        """Resize Masks.

        Args:
            masks (List): list of raw np.ndarray masks
            dets (np.ndarray): detections including labels, scores, and boxes
            shape (List[int]): original full-res image shape
        """
        for i, (det, mask) in enumerate(zip(dets, masks)):
            masks[i] = self.model.segm_postprocess(det[2:], mask, *shape[:-1])

    def predict_tile(
        self,
        image: np.ndarray,
        coord: List[int],
        masks: List[np.ndarray],
        return_features=False,
    ):
        """Predict on single tile.

        Args:
            image (np.ndarray): full-res image
            coord (List): tile coordinates
            masks (List): list of raw np.ndarray masks
            return_features (bool, optional): return saliency map and feature vector if set to true. Defaults to False.

        Returns:
            features: saliency map and feature vector
            output: single tile prediction
        """
        features = (None, None)
        offset_x, offset_y, tile_dict, tile_meta = self.preprocess_tile(image, coord)
        raw_predictions = self.model.infer_sync(tile_dict)
        output = self.model.postprocess(raw_predictions, tile_meta)
        output = self.postprocess_tile(output, offset_x, offset_y, masks)
        if return_features:
            if "feature_vector" in raw_predictions or "saliency_map" in raw_predictions:
                features = (
                    raw_predictions["feature_vector"].reshape(-1),
                    raw_predictions["saliency_map"][0],
                )
        return features, output

    def postprocess_tile(
        self,
        output: Union[List, Tuple],
        offset_x: int,
        offset_y: int,
        masks: List,
    ):
        """Postprocess tile predictions.

        Args:
            output (Union[List, Tuple]): predictions
            offset_x (int): tile offset x value
            offset_y (int): tile offset y value
            masks (List): list of raw np.ndarray mask

        Returns:
            output: processed tile prediction
        """
        if self.segm:
            tile_scores, tile_labels, tile_boxes, tile_masks = output
            tile_boxes += np.tile([offset_x, offset_y], 2)
            out = np.concatenate(
                (
                    tile_labels[:, np.newaxis],
                    tile_scores[:, np.newaxis],
                    tile_boxes,
                ),
                -1,
            )
            masks.extend(tile_masks)
        else:
            assert isinstance(output, list)
            out = detection2array(output)
            out[:, 2:] += np.tile([offset_x, offset_y], 2)
        return out

    def preprocess_tile(self, image: np.ndarray, coord: List[int]):
        """Preprocess Tile by cropping.

        Args:
            image (np.ndarray): full-res image
            coord (List): tile coordinates

        Returns:
            _type_: _description_
        """
        x1, y1, x2, y2 = coord
        tile_dict, tile_meta = self.model.preprocess(image[y1:y2, x1:x2])
        if self.segm:
            tile_meta["resize_mask"] = False
        return x1, y1, tile_dict, tile_meta

    @staticmethod
    def detection2tuple(detections: np.ndarray):
        """_summary_.

        Args:
            detections (np.ndarray): _description_

        Returns:
            scores (np.ndarray): scores between 0-1
            labels (np.ndarray): label indices
            boxes (np.ndarray): boxes
        """
        labels = detections[:, 0]
        scores = detections[:, 1]
        boxes = detections[:, 2:]
        return scores, labels, boxes
