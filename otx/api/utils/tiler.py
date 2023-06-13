"""Tiling Module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import cv2
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from openvino.model_zoo.model_api.models import Model

from otx.api.utils.async_pipeline import OTXDetectionAsyncPipeline
from otx.api.utils.detection_utils import detection2array
from otx.api.utils.nms import multiclass_nms


class Tiler:
    """Tile Image into (non)overlapping Patches. Images are tiled in order to efficiently process large images.

    Args:
        tile_size: Tile dimension for each patch
        overlap: Overlap between adjacent tile
        max_number: max number of prediction per image
        detector: OpenVINO adaptor model
        classifier: Tile classifier OpenVINO adaptor model
        segm: enable instance segmentation mask output
        mode: async or sync mode
    """

    def __init__(
        self,
        tile_size: int,
        overlap: float,
        max_number: int,
        detector: Any,
        classifier: Model,
        segm: bool = False,
        mode: str = "async",
    ):  # pylint: disable=too-many-arguments
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_number = max_number
        self.model = detector
        self.classifier = classifier
        self.segm = segm
        if self.segm:
            self.model.disable_mask_resizing()
        if mode == "async":
            self.async_pipeline = OTXDetectionAsyncPipeline(self.model)

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
            range(0, width, int(self.tile_size * (1 - self.overlap))),
            range(0, height, int(self.tile_size * (1 - self.overlap))),
        ):
            x2 = min(loc_j + self.tile_size, width)
            y2 = min(loc_i + self.tile_size, height)
            coords.append([loc_j, loc_i, x2, y2])
        print(f"------------------------> Num tiles: {len(coords)}")
        print(f"------------------------> {height}x{width} ~ {self.tile_size}")
        return coords

    def filter_tiles_by_objectness(
        self, image: np.ndarray, tile_coords: List[List[int]], confidence_threshold: float = 0.35
    ):
        """Filter tiles by objectness score by running tile classifier.

        Args:
            image (np.ndarray): full size image
            tile_coords (List[List[int]]): tile coordinates

        Returns:
            keep_coords: tile coordinates to keep
        """
        keep_coords = []
        for i, coord in enumerate(tile_coords):
            tile_img = self.crop_tile(image, coord)
            tile_dict, _ = self.model.preprocess(tile_img)
            objectness_score = self.classifier.infer_sync(tile_dict)
            if i == 0 or objectness_score["tile_prob"] > confidence_threshold:
                keep_coords.append(coord)
        return keep_coords

    def predict(self, image: np.ndarray, mode: str = "async"):
        """Predict by cropping full image to tiles.

        Args:
            image (np.ndarray): full size image

        Returns:
            detection: prediction results
            features: saliency map and feature vector
        """
        tile_coords = self.tile(image)
        if isinstance(self.classifier, Model):
            tile_coords = self.filter_tiles_by_objectness(image, tile_coords)

        if mode == "sync":
            return self.predict_sync(image, tile_coords)
        return self.predict_async(image, tile_coords)

    def predict_sync(self, image: np.ndarray, tile_coords: List[List[int]]):
        """Predict by cropping full image to tiles synchronously.

        Args:
            image (np.ndarray): full size image
            tile_coords (List[List[int]]): tile coordinates

        Returns:
            detection: prediction results
            features: saliency map and feature vector
        """
        features = []
        tile_results = []

        for coord in tile_coords:
            tile_img = self.crop_tile(image, coord)
            tile_dict, tile_meta = self.model.preprocess(tile_img)
            raw_predictions = self.model.infer_sync(tile_dict)
            predictions = self.model.postprocess(raw_predictions, tile_meta)
            tile_result = self.postprocess_tile(predictions, *coord[:2])
            # cache each tile feature vector and saliency map
            if "feature_vector" in raw_predictions or "saliency_map" in raw_predictions:
                tile_meta.update({"coord": coord})
                features.append(
                    (
                        (
                            copy.deepcopy(raw_predictions["feature_vector"].reshape(-1)),
                            copy.deepcopy(raw_predictions["saliency_map"][0]),
                        ),
                        tile_meta,
                    )
                )

            tile_results.append(tile_result)

        results = self.merge_results(tile_results, image.shape)
        merged_tile_features = self.merge_features(features)
        return results, merged_tile_features

    def predict_async(self, image: np.ndarray, tile_coords: List[List[int]]):
        """Predict by cropping full image to tiles asynchronously.

        Args:
            image (np.ndarray): full size image
            tile_coords (List[List[int]]): tile coordinates

        Returns:
            detection: prediction results
            features: saliency map and feature vector
        """
        num_tiles = len(tile_coords)

        processed_tiles = 0
        tile_results = []
        features = []
        for i, coord in enumerate(tile_coords):
            pred = self.async_pipeline.get_result(processed_tiles)
            while pred:
                tile_prediction, meta, feats = pred
                features.append((feats, meta))
                tile_result = self.postprocess_tile(tile_prediction, *meta["coord"][:2])
                tile_results.append(tile_result)
                processed_tiles += 1
                pred = self.async_pipeline.get_result(processed_tiles)
            self.async_pipeline.submit_data(self.crop_tile(image, coord), i, {"coord": coord, "tile_i": i})

        self.async_pipeline.await_all()
        for j in range(processed_tiles, num_tiles):
            tile_prediction, meta, feats = self.async_pipeline.get_result(j)
            features.append((feats, meta))
            tile_result = self.postprocess_tile(tile_prediction, *meta["coord"][:2])
            tile_results.append(tile_result)
        assert j == num_tiles - 1, "Number of tiles processed does not match number of tiles"
        return self.merge_results(tile_results, image.shape), self.merge_features(features)

    def postprocess_tile(self, predictions: Union[List, Tuple], offset_x: int, offset_y: int) -> Dict[str, List]:
        """Postprocess single tile prediction.

        Args:
            predictions (Union[List, Tuple]): predictions from model
            offset_x (int): tile offset in x direction
            offset_y (int): tile offset in y direction

        Returns:
            Dict[str, List]: postprocessed predictions - bboxes and masks
        """
        output_dict: dict = {"bboxes": [], "masks": []}
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
            detections (np.ndarray): prediction results in numpy array

        Returns:
            scores (np.ndarray): scores between 0-1
            labels (np.ndarray): label indices
            boxes (np.ndarray): boxes
        """
        labels = detections[:, 0]
        scores = detections[:, 1]
        boxes = detections[:, 2:]
        return scores, labels, boxes

    def merge_results(self, results: List[Dict], shape: List[int]):
        """Merge results from tiles.

        Args:
            results (List[Dict]): list of tile results
            shape (List[int]): original full-res image shape
        """

        detections = np.empty((0, 6), dtype=np.float32)
        masks = []
        for result in results:
            if len(result["bboxes"]):
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

    def merge_features(self, features: List) -> Union[Tuple[None, None], List[np.ndarray]]:
        """Merge tile-level feature vectors to image-level features.

        Args:
            features: tile-level features.

        Returns:
            merged_features (list[np.ndarray]): Merged features for entire image.
        """
        if len(features) == 0:
            return (None, None)
        image_vector = self.merge_vectors(features)
        image_saliency_map = self.merge_maps(features)
        return image_vector, image_saliency_map

    def merge_vectors(self, features: List) -> np.ndarray:
        """Merge tile-level feature vectors to image-level feature vector.

        Args:
            features: tile-level features.

        Returns:
            merged_vectors (np.ndarray): Merged vectors for entire image.
        """

        vectors = [vector for (vector, _), _ in features]
        return np.average(vectors, axis=0)

    def merge_maps(self, features: List) -> np.ndarray:
        """Merge tile-level saliency maps to image-level saliency map.

        Args:
            features: tile-level features.

        Returns:
            merged_maps (np.ndarray): Merged saliency maps for entire image.
        """

        (_, image_saliency_map), image_meta = features[0]
        dtype = image_saliency_map[0][0].dtype
        feat_classes, feat_h, feat_w = image_saliency_map.shape
        ratio = np.array([feat_h, feat_w]) / self.tile_size
        image_h, image_w, _ = image_meta["original_shape"]

        merged_maps_size = np.array(
            (feat_classes, image_h * ratio[0], image_w * ratio[1]),
            dtype=np.uint8,
        )
        merged_map = np.zeros(merged_maps_size, dtype=dtype)

        # resize the feature map for whole image to add it to merged saliency maps
        image_saliency_map = np.array(
            [cv2.resize(class_sal_map, merged_maps_size[1:]) for class_sal_map in image_saliency_map]
        )

        for (_, saliency_map), meta in features[1:]:
            x_1, y_1, x_2, y_2 = (meta["coord"] * np.repeat(ratio, 2)).astype(np.uint8)
            c = saliency_map.shape[0]
            w, h = x_2 - x_1, y_2 - y_1

            for ci in range(c):
                for hi in range(h):
                    for wi in range(w):
                        map_pixel = saliency_map[ci, hi, wi]
                        if merged_map[ci, y_1 + hi, x_1 + wi] != 0:
                            merged_map[ci, y_1 + hi, x_1 + wi] = 0.5 * (map_pixel + merged_map[ci, y_1 + hi, x_1 + wi])
                        else:
                            merged_map[ci, y_1 + hi, x_1 + wi] = map_pixel

        merged_map += (0.5 * image_saliency_map).astype(dtype)
        min_soft_score = np.min(merged_map, axis=(1, 2)).reshape(-1, 1, 1)
        # make merged_map distribution positive to perform non-linear normalization y=x**1.5
        merged_map = (merged_map - min_soft_score) ** 1.5
        max_soft_score = np.max(merged_map, axis=(1, 2)).reshape(-1, 1, 1)
        merged_map = 255.0 / (max_soft_score + 1e-12) * merged_map
        merged_map = np.uint8(np.floor(merged_map))

        return merged_map

    def resize_masks(self, masks: List, dets: np.ndarray, shape: List[int]):
        """Resize Masks.

        Args:
            masks (List): list of raw np.ndarray masks
            dets (np.ndarray): detections including labels, scores, and boxes
            shape (List[int]): original full-res image shape
        """
        for i, (det, mask) in enumerate(zip(dets, masks)):
            masks[i] = self.model.segm_postprocess(det[2:], mask, *shape[:-1])
