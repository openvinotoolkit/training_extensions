"""Tiling Module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import cv2
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from openvino.model_api.models import Model, ImageModel

from openvino.model_api.models.utils import DetectionResult

from otx.api.utils.async_pipeline import OTXDetectionAsyncPipeline
from otx.api.utils.detection_utils import detection2array
from otx.api.utils.nms import multiclass_nms
from otx.api.utils.dataset_utils import non_linear_normalization


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
        detector: Model,
        classifier: Optional[ImageModel] = None,
        segm: bool = False,
        mode: str = "async",
        num_classes: int = 0,
    ):  # pylint: disable=too-many-arguments
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_number = max_number
        self.model = detector
        self.classifier = classifier
        # needed to create saliency maps for IRs for Mask RCNN
        self.num_classes = num_classes
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
        if self.classifier is not None:
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
                        (raw_predictions["feature_vector"].reshape(-1), raw_predictions["saliency_map"][0]),
                        tile_meta,
                    )
                )

            tile_results.append(tile_result)

        merged_results = self.merge_results(tile_results, image.shape)
        merged_features = self.merge_features(features, merged_results)
        return merged_results, merged_features

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
                if isinstance(feats[0], np.ndarray):
                    features.append((feats, meta))
                tile_result = self.postprocess_tile(tile_prediction, *meta["coord"][:2])
                tile_results.append(tile_result)
                processed_tiles += 1
                pred = self.async_pipeline.get_result(processed_tiles)
            self.async_pipeline.submit_data(self.crop_tile(image, coord), i, {"coord": coord, "tile_i": i})

        self.async_pipeline.await_all()
        for j in range(processed_tiles, num_tiles):
            tile_prediction, meta, feats = self.async_pipeline.get_result(j)
            if isinstance(feats[0], np.ndarray):
                features.append((feats, meta))
            tile_result = self.postprocess_tile(tile_prediction, *meta["coord"][:2])
            tile_results.append(tile_result)
        assert j == num_tiles - 1, "Number of tiles processed does not match number of tiles"
        merged_results = self.merge_results(tile_results, image.shape)
        merged_features = self.merge_features(features, merged_results)
        return merged_results, merged_features

    def postprocess_tile(self, predictions: DetectionResult, offset_x: int, offset_y: int) -> Dict[str, List]:
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
            assert isinstance(predictions.objects, list)
            out = detection2array(predictions.objects)
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

    def merge_features(
        self, features: List, predictions: Union[Tuple, np.ndarray]
    ) -> Union[Tuple[None, None], List[np.ndarray]]:
        """Merge tile-level feature vectors to image-level features.

        Args:
            features: tile-level features.
            predictions: predictions with masks for whole image.

        Returns:
            image_vector (np.ndarray): Merged feature vector for entire image.
            image_saliency_map (List): Merged saliency map for entire image
        """
        if len(features) == 0:
            return (None, None)
        image_vector = self.merge_vectors(features)

        (_, image_saliency_map), _ = features[0]
        if isinstance(image_saliency_map, np.ndarray):
            image_saliency_map = self.merge_maps(features)
        else:
            # if saliency maps weren't return from hook (Mask RCNN case)
            image_saliency_map = self.get_tiling_saliency_map_from_segm_masks(predictions)

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
            features: tile-level features ((vector, map: np.array), tile_meta).
            Each saliency map is a list of maps for each detected class or None if class wasn't detected.

        Returns:
            merged_maps (np.ndarray): Merged saliency maps for entire image.
        """
        (_, image_saliency_map), image_meta = features[0]

        num_classes, feat_h, feat_w = image_saliency_map.shape
        dtype = image_saliency_map[0][0].dtype

        image_h, image_w, _ = image_meta["original_shape"]
        ratio = np.array([feat_h / min(self.tile_size, image_h), feat_w / min(self.tile_size, image_w)])

        image_map_h = int(image_h * ratio[0])
        image_map_w = int(image_w * ratio[1])
        # happens because of the bug then tile_size for IR in a few times more than original image
        if image_map_h == 0 or image_map_w == 0:
            return [None] * num_classes
        merged_map = [np.zeros((image_map_h, image_map_w)) for _ in range(num_classes)]

        for (_, saliency_map), meta in features[1:]:
            x_1, y_1, x_2, y_2 = meta["coord"]
            y_1, x_1 = ((y_1, x_1) * ratio).astype(np.uint16)
            y_2, x_2 = ((y_2, x_2) * ratio).astype(np.uint16)

            map_h, map_w = saliency_map[0].shape
            # resize feature map if it got from the tile which width and height is less the tile_size
            if (map_h > y_2 - y_1 > 0) and (map_w > x_2 - x_1 > 0):
                saliency_map = np.array([cv2.resize(cls_map, (x_2 - x_1, y_2 - y_1)) for cls_map in saliency_map])
            # cut the rest of the feature map that went out of the image borders
            map_h, map_w = y_2 - y_1, x_2 - x_1

            for ci, hi, wi in [(c_, h_, w_) for c_ in range(num_classes) for h_ in range(map_h) for w_ in range(map_w)]:
                map_pixel = saliency_map[ci, hi, wi]
                # on tile overlap add 0.5 value of each tile
                if merged_map[ci][y_1 + hi, x_1 + wi] != 0:
                    merged_map[ci][y_1 + hi, x_1 + wi] = 0.5 * (map_pixel + merged_map[ci][y_1 + hi, x_1 + wi])
                else:
                    merged_map[ci][y_1 + hi, x_1 + wi] = map_pixel

        for class_idx in range(num_classes):
            image_map_cls = image_saliency_map[class_idx]
            # resize the feature map for whole image to add it to merged saliency maps
            image_map_cls = cv2.resize(image_map_cls, (image_map_w, image_map_h))
            merged_map[class_idx] += (0.5 * image_map_cls).astype(dtype)
            merged_map[class_idx] = non_linear_normalization(merged_map[class_idx])
        return merged_map

    def get_tiling_saliency_map_from_segm_masks(self, detections: Union[Tuple, np.ndarray]) -> List:
        """Post process function for saliency map of OTX MaskRCNN model for tiling."""

        # No detection case
        if isinstance(detections, np.ndarray) and detections.size == 0:
            return [None]
        # Exportable demo case
        if self.num_classes == 0:
            return [None]

        classes = [int(cls) - 1 for cls in detections[1]]
        saliency_maps: List = [None for _ in range(self.num_classes)]
        scores = detections[0].reshape(-1, 1, 1)
        masks = detections[3]
        weighted_masks = masks * scores
        for mask, cls in zip(weighted_masks, classes):
            if saliency_maps[cls] is None:
                saliency_maps[cls] = [mask]
            else:
                saliency_maps[cls].append(mask)
        saliency_maps = self._merge_and_normalize(saliency_maps, self.num_classes)
        return saliency_maps

    @staticmethod
    def _merge_and_normalize(saliency_maps: List, num_classes: int) -> List:
        for i in range(num_classes):
            if saliency_maps[i] is not None:
                # combine masks for all objects within one class
                saliency_maps[i] = np.max(np.array(saliency_maps[i]), axis=0)

        for i in range(num_classes):
            per_class_map = saliency_maps[i]
            if per_class_map is not None:
                max_values = np.max(per_class_map)
                per_class_map = 255 * (per_class_map) / (max_values + 1e-12)
                per_class_map = per_class_map.astype(np.uint8)
                saliency_maps[i] = per_class_map
        return saliency_maps

    def resize_masks(self, masks: List, dets: np.ndarray, shape: List[int]):
        """Resize Masks.

        Args:
            masks (List): list of raw np.ndarray masks
            dets (np.ndarray): detections including labels, scores, and boxes
            shape (List[int]): original full-res image shape
        """
        for i, (det, mask) in enumerate(zip(dets, masks)):
            masks[i] = self.model.segm_postprocess(det[2:], mask, *shape[:-1])
