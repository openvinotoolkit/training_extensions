"""Tiling Module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from otx.api.utils.async_pipeline import OTXAsyncPipeline
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
        if self.segm:
            self.model.disable_mask_resizing()
        self.async_pipeline = OTXAsyncPipeline(self.model)

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

    def predict_async(self, image: np.ndarray):
        """ Predict by cropping full image to tiles asynchronously.

        Args:
            image (np.ndarray): full size image

        Returns:
            detection: prediction results
            features: saliency map and feature vector
        """

        processed_frame = 0
        results = []
        features = (None, None)
        for i, coord in enumerate(self.tile(image)):
            pred = self.async_pipeline.get_result(processed_frame)
            while pred:
                predictions, meta, feats = pred
                if meta["tile_i"] == 0:
                    features = copy.deepcopy(feats)
                result = self.postprocess_tile(predictions, *meta["coord"])
                results.append(result)
                processed_frame += 1
                pred = self.async_pipeline.get_result(processed_frame)
            self.async_pipeline.submit_data(self.crop_tile(image, coord), i, {"coord": coord, "tile_i": i})

        self.async_pipeline.await_all()
        for j in range(processed_frame, i):
            predictions, meta, feats = self.async_pipeline.get_result(j)
            result = self.postprocess_tile(predictions, *meta["coord"])
            results.append(result)
        results = self.merge_results(results, image.shape)
        return results, features

    def predict(self, image: np.ndarray):
        """Predict by cropping full image to tiles.

        Args:
            image (np.ndarray): full size image

        Returns:
            detection: prediction results
            features: saliency map and feature vector
        """
        features = (None, None)
        results = []
        for i, coord in enumerate(self.tile(image)):
            result, feats = self.predict_tile(image, coord, i == 0)
            results.append(result)
            # cache full image feature vector and saliency map at 0 index
            if i == 0:
                features = copy.deepcopy(feats)

        results = self.merge_results(results, image.shape)
        return results, features

    def merge_results(self, results: List[Dict], shape: List[int]):
        """Merge results from tiles.

        Args:
            results (List[Dict]): _description_
            shape (List[int]): original full-res image shape
        """

        detections = np.empty((0, 6), dtype=np.float32)
        masks = []
        for result in results:
            detections = np.concatenate((detections, result["bboxes"]))
            if self.segm:
                masks.extend(result["masks"])

        if np.prod(detections.shape):
            detections, keep = multiclass_nms(detections, max_num=self.max_number)
            if self.segm:
                masks = [masks[keep_idx] for keep_idx in keep]
                self.resize_masks(masks, detections, shape)
                detections = *Tiler.detection2tuple(detections), masks
        return detections

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
        return_features=False,
    ) -> Tuple[Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]], Dict]:
        """Predict on single tile.

        Args:
            image (np.ndarray): full-res image
            coord (List): tile coordinates
            return_features (bool, optional): return saliency map and feature vector if set to true. Defaults to False.

        Returns:
            features: saliency map and feature vector
            output_dict: single tile prediction
        """
        features = (None, None)
        tile_img = self.crop_tile(image, coord)
        tile_dict, tile_meta = self.model.preprocess(tile_img)
        raw_predictions = self.model.infer_sync(tile_dict)
        predictions = self.model.postprocess(raw_predictions, tile_meta)
        output_dict = self.postprocess_tile(predictions, *coord)
        if return_features:
            if "feature_vector" in raw_predictions or "saliency_map" in raw_predictions:
                features = (
                    raw_predictions["feature_vector"].reshape(-1),
                    raw_predictions["saliency_map"][0],
                )
        return output_dict, features

    def postprocess_tile(self, predictions: Union[List, Tuple], offset_x: int, offset_y: int, *args) -> Dict[str, List]:
        """ Postprocess single tile prediction.

        Args:
            predictions (Union[List, Tuple]): predictions from model
            offset_x (int): tile offset in x direction
            offset_y (int): tile offset in y direction

        Returns:
            Dict[str, List]: postprocessed predictions - bboxes and masks
        """
        output_dict = {"bboxes": [], "masks": []}
        if self.segm:
            tile_scores, tile_labels, tile_boxes, tile_masks = predictions
            tile_boxes += np.tile([offset_x, offset_y], 2)
            out = np.concatenate(
                (
                    tile_labels[:, np.newaxis],
                    tile_scores[:, np.newaxis],
                    tile_boxes,
                ),
                -1,
            )
            output_dict["masks"] = tile_masks
        else:
            assert isinstance(predictions, list)
            out = detection2array(predictions)
            out[:, 2:] += np.tile([offset_x, offset_y], 2)
        output_dict["bboxes"] = out
        return output_dict

    def crop_tile(self, image: np.ndarray, coord: List[int]) -> np.ndarray:
        """Crop tile from full image.

        Args:
            image (np.ndarray): full-res image
            coord (List): tile coordinates

        Returns:
            np.ndarray: cropped tile
        """
        x1, y1, x2, y2 = coord
        return image[y1:y2, x1:x2]

    @staticmethod
    def detection2tuple(detections: np.ndarray):
        """Convert detection to tuple.

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
