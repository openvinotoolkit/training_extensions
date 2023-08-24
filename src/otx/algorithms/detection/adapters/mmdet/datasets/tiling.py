"""Tiling for detection and instance segmentation task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import uuid
from itertools import product
from random import sample
from time import time
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
from mmcv.ops import nms
from mmdet.core import BitmapMasks, bbox2result
from tqdm import tqdm

from otx.api.utils.dataset_utils import non_linear_normalization


def timeit(func) -> Callable:
    """Decorator to measure time of function execution.

    Args:
        func: Function to be the target for measuring.

    Returns:
        Callable function with time measurement.
    """

    def wrapper(*args, **kwargs):
        begin = time()
        result = func(*args, **kwargs)
        print(f"\n==== {func.__name__}: {time() - begin} sec ====\n")
        return result

    return wrapper


# pylint: disable=too-many-instance-attributes, too-many-arguments
class Tile:
    """Tile and merge datasets.

    Args:
        dataset (CustomDataset): the dataset to be tiled.
        tile_size (int): the length of side of each tile. Defaults to 400
        overlap (float, optional): ratio of each tile to overlap with each of
            the tiles in its 4-neighborhood. Defaults to 0.2.
        min_area_ratio (float, optional): the minimum overlap area ratio
            between a tiled image and its annotations. Ground-truth box is
            discarded if the overlap area is less than this value.
            Defaults to 0.9.
        iou_threshold (float, optional): IoU threshold to be used to suppress
            boxes in tiles' overlap areas. Defaults to 0.45.
        max_per_img (int, optional): if there are more than max_per_img bboxes
            after NMS, only top max_per_img will be kept. Defaults to 1500.
        max_annotation(int, optional): Limit the number of ground truth by
            randomly select 5000 due to RAM OOM.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests. Defaults to True.
        nproc (int, optional): Processes used for processing masks. Default: 4.
        sampling_ratio (float): Ratio for sampling entire tile dataset. Default: 1.0.(No sample)
        include_full_img (bool): Whether to include full-size image for inference or training. Default: False.
    """

    def __init__(
        self,
        dataset,
        pipeline,
        tile_size: int = 400,
        overlap: float = 0.2,
        min_area_ratio: float = 0.9,
        iou_threshold: float = 0.45,
        max_per_img: int = 1500,
        max_annotation: int = 2000,
        filter_empty_gt: bool = True,
        nproc: int = 2,
        sampling_ratio: float = 1.0,
        include_full_img: bool = False,
    ):
        self.min_area_ratio = min_area_ratio
        self.filter_empty_gt = filter_empty_gt
        self.iou_threshold = iou_threshold
        self.max_per_img = max_per_img
        self.max_annotation = max_annotation
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.num_images = len(dataset)
        self.num_classes = len(dataset.CLASSES)
        self.CLASSES = dataset.CLASSES  # pylint: disable=invalid-name
        self.nproc = nproc
        self.img2fp32 = False
        for p in pipeline:
            if p.type == "PhotoMetricDistortion":
                self.img2fp32 = True
                break

        self.dataset = dataset
        self.tiles_all, self.cached_results = self.gen_tile_ann(include_full_img)
        self.sample_num = max(int(len(self.tiles_all) * sampling_ratio), 1)
        if sampling_ratio < 1.0:
            self.tiles = sample(self.tiles_all, self.sample_num)
        else:
            self.tiles = self.tiles_all

    @timeit
    def gen_tile_ann(self, include_full_img) -> Tuple[List[Dict], List[Dict]]:
        """Generate tile annotations and cache the original image-level annotations.

        Returns:
            tiles: a list of tile annotations with some other useful information for data pipeline.
            cache_result: a list of original image-level annotations.
            include_full_img: whether to include full-size image for inference or training.
        """
        tiles = []
        cache_result = []
        for result in tqdm(self.dataset, desc="Loading dataset annotations..."):
            cache_result.append(result)

        pbar = tqdm(total=len(self.dataset) * 2, desc="Generating tile annotations...")
        for idx, result in enumerate(cache_result):
            if include_full_img:
                tiles.append(self.gen_single_img(result, dataset_idx=idx))
            pbar.update(1)

        for idx, result in enumerate(cache_result):
            tiles.extend(self.gen_tiles_single_img(result, dataset_idx=idx))
            pbar.update(1)
        return tiles, cache_result

    def gen_single_img(self, result: Dict, dataset_idx: int) -> Dict:
        """Add full-size image for inference or training.

        Args:
            result (Dict): the original image-level result (i.e. the original image annotation)
            dataset_idx (int): the image index this tile belongs to

        Returns:
            Dict: annotation with some other useful information for data pipeline.
        """
        result["full_res_image"] = True
        result["tile_box"] = (0, 0, result["img_shape"][1], result["img_shape"][0])
        result["dataset_idx"] = dataset_idx
        result["original_shape_"] = result["img_shape"]
        result["uuid"] = str(uuid.uuid4())
        result["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
        result["gt_labels"] = np.array([], dtype=int)
        result["gt_masks"] = []

        # Limit the number of ground truth by randomly select 5000 get due to RAM OOM
        if "gt_masks" in result and len(result["gt_masks"]) > self.max_annotation:
            indices = np.random.choice(len(result["gt_bboxes"]), size=self.max_annotation, replace=False)
            result["gt_bboxes"] = result["gt_bboxes"][indices]
            result["gt_labels"] = result["gt_labels"][indices]
            result["gt_masks"] = result["gt_masks"][indices]
        return result

    # pylint: disable=too-many-locals
    def gen_tiles_single_img(self, result: Dict, dataset_idx: int) -> List[Dict]:
        """Generate tile annotation for a single image.

        Args:
            result (Dict): the original image-level result (i.e. the original image annotation)
            dataset_idx (int): the image index this tile belongs to

        Returns:
            List[Dict]: a list of tile annotation with some other useful information for data pipeline.
        """
        tile_list = []
        gt_bboxes = result.get("gt_bboxes", np.zeros((0, 4), dtype=np.float32))
        gt_masks = result.get("gt_masks", None)
        gt_bboxes_ignore = result.get("gt_bboxes_ignore", np.zeros((0, 4), dtype=np.float32))
        gt_labels = result.get("gt_labels", np.array([], dtype=np.int64))
        img_shape = result.get("img_shape")
        height, width = img_shape[:2]
        _tile = self.prepare_result(result)

        num_patches_h = (height + self.stride - 1) // self.stride
        num_patches_w = (width + self.stride - 1) // self.stride
        for (_, _), (loc_i, loc_j) in zip(
            product(range(num_patches_h), range(num_patches_w)),
            product(
                range(0, height, self.stride),
                range(0, width, self.stride),
            ),
        ):
            x_1 = loc_j
            x_2 = min(loc_j + self.tile_size, width)
            y_1 = loc_i
            y_2 = min(loc_i + self.tile_size, height)
            tile = copy.deepcopy(_tile)
            tile["full_res_image"] = False
            tile["original_shape_"] = img_shape
            tile["ori_shape"] = (y_2 - y_1, x_2 - x_1, 3)
            tile["img_shape"] = tile["ori_shape"]
            tile["tile_box"] = (x_1, y_1, x_2, y_2)
            tile["dataset_idx"] = dataset_idx
            tile["gt_bboxes_ignore"] = gt_bboxes_ignore
            tile["uuid"] = str(uuid.uuid4())
            self.tile_ann_assignment(tile, np.array([[x_1, y_1, x_2, y_2]]), gt_bboxes, gt_masks, gt_labels)
            # filter empty ground truth
            if self.filter_empty_gt and len(tile["gt_labels"]) == 0:
                continue
            tile_list.append(tile)
        if dataset_idx == 0:
            print(f"image: {height}x{width} ~ tile_size: {self.tile_size}")
            print(f"{num_patches_h}x{num_patches_w} tiles -> {len(tile_list)} tiles after filtering")
        return tile_list

    def prepare_result(self, result: Dict) -> Dict:
        """Prepare results dict for pipeline.

        Args:
            result (Dict): original image-level result for a tile

        Returns:
            Dict: result template with useful information for data pipeline.
        """
        result_template = dict(
            ori_filename=result["ori_filename"],
            filename=result["filename"],
            bbox_fields=result["bbox_fields"],
            mask_fields=result["mask_fields"],
            seg_fields=result["seg_fields"],
            img_fields=result["img_fields"],
        )
        return result_template

    def tile_ann_assignment(
        self,
        tile_result: Dict,
        tile_box: np.ndarray,
        gt_bboxes: np.ndarray,
        gt_masks: BitmapMasks,
        gt_labels: np.ndarray,
    ):
        """Assign new annotation to this tile.

        Ground-truth is discarded if the overlap with this tile is lower than
        min_area_ratio.

        Args:
            tile_result (Dict): the tile-level result (i.e. the tile annotation)
            tile_box (np.ndarray): the coordinate for this tile box (i.e. the tile coordinate relative to the image)
            gt_bboxes (np.ndarray): the original image-level boxes
            gt_masks (BitmapMasks): the original image-level masks
            gt_labels (np.ndarray): the original image-level labels
        """
        x_1, y_1 = tile_box[0][:2]
        matched_indices = self.tile_boxes_overlap(tile_box, gt_bboxes)

        if len(matched_indices):
            tile_lables = gt_labels[matched_indices][:]
            tile_bboxes = gt_bboxes[matched_indices][:]
            tile_bboxes[:, 0] -= x_1
            tile_bboxes[:, 1] -= y_1
            tile_bboxes[:, 2] -= x_1
            tile_bboxes[:, 3] -= y_1
            tile_bboxes[:, 0] = np.maximum(0, tile_bboxes[:, 0])
            tile_bboxes[:, 1] = np.maximum(0, tile_bboxes[:, 1])
            tile_bboxes[:, 2] = np.minimum(self.tile_size, tile_bboxes[:, 2])
            tile_bboxes[:, 3] = np.minimum(self.tile_size, tile_bboxes[:, 3])
            tile_result["gt_bboxes"] = tile_bboxes
            tile_result["gt_labels"] = tile_lables
            tile_result["gt_masks"] = gt_masks[matched_indices].crop(tile_box[0]) if gt_masks is not None else []
        else:
            tile_result.pop("bbox_fields")
            tile_result.pop("mask_fields")
            tile_result.pop("seg_fields")
            tile_result.pop("img_fields")
            tile_result["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
            tile_result["gt_labels"] = np.array([], dtype=int)
            tile_result["gt_masks"] = []

        if gt_masks is None:
            tile_result.pop("gt_masks")

    def tile_boxes_overlap(self, tile_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute overlapping ratio over boxes.

        Args:
            tile_box (np.ndarray): box in shape (1, 4).
            boxes (np.ndarray): boxes in shape (N, 4).

        Returns:
            np.ndarray: matched indices.
        """
        x1, y1, x2, y2 = tile_box[0]
        match_indices = (boxes[:, 0] > x1) & (boxes[:, 1] > y1) & (boxes[:, 2] < x2) & (boxes[:, 3] < y2)
        match_indices = np.argwhere(match_indices == 1).flatten()
        return match_indices

    def multiclass_nms(
        self, boxes: np.ndarray, scores: np.ndarray, idxs: np.ndarray, iou_threshold: float, max_num: int
    ):
        """NMS for multi-class bboxes.

        Args:
            boxes (np.ndarray):  boxes in shape (N, 4).
            scores (np.ndarray): scores in shape (N, ).
            idxs (np.ndarray):  each index value correspond to a bbox cluster,
                and NMS will not be applied between elements of different idxs,
                shape (N, ).
            iou_threshold (float): IoU threshold to be used to suppress boxes
                in tiles' overlap areas.
            max_num (int): if there are more than max_per_img bboxes after
                NMS, only top max_per_img will be kept.

        Returns:
            tuple: tuple: kept dets and indice.
        """
        if len(boxes) == 0:
            return None, []
        max_coordinate = boxes.max()
        offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        dets, keep = nms(boxes_for_nms, scores, iou_threshold)
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]
        return dets, keep

    def tile_nms(
        self,
        bbox_results: List[np.ndarray],
        mask_results: List[List],
        label_results: List[np.ndarray],
        iou_threshold: float,
        max_per_img: int,
        detection: bool,
    ):
        """NMS after aggregation suppressing duplicate boxes in tile-overlap areas.

        Args:
            bbox_results (List[List]): image-level box prediction
            mask_results (List[np.ndarray]): image-level mask prediction
            label_results (List[List]): image-level label prediction
            iou_threshold (float): IoU threshold to be used to suppress boxes in tiles' overlap areas.
            max_per_img (int): if there are more than max_per_img bboxes after NMS, only top max_per_img will be kept.
            detection (bool): whether it is a detection task
        """
        assert len(bbox_results) == len(mask_results) == len(label_results)
        for i, result in enumerate(zip(bbox_results, mask_results, label_results)):
            score_bboxes, masks, labels = result
            bboxes = score_bboxes[:, :4]
            scores = np.ascontiguousarray(score_bboxes[:, 4])
            _, keep_indices = self.multiclass_nms(
                bboxes, scores, labels, iou_threshold=iou_threshold, max_num=max_per_img
            )

            bboxes = bboxes[keep_indices]
            labels = labels[keep_indices]
            scores = scores[keep_indices]
            bbox_results[i] = bbox2result(np.concatenate([bboxes, scores[:, None]], -1), labels, self.num_classes)

            if not detection:
                masks = np.array([masks[keep_idx] for keep_idx in keep_indices])
                mask_results[i] = [list(masks[labels == i]) for i in range(self.num_classes)]

    def __len__(self):
        """Total number of tiles."""
        return len(self.tiles)

    def __getitem__(self, idx):
        """Get training/test tile.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data.
        """
        result = copy.deepcopy(self.tiles[idx])
        dataset_idx = result["dataset_idx"]
        x_1, y_1, x_2, y_2 = result["tile_box"]
        ori_img = self.cached_results[dataset_idx]["img"]
        cropped_tile = ori_img[y_1:y_2, x_1:x_2, :]
        if self.img2fp32:
            cropped_tile = cropped_tile.astype(np.float32)
        result["img"] = cropped_tile
        return result

    # pylint: disable=too-many-locals
    @timeit
    def merge(self, results: List[List]) -> Union[List[Tuple[np.ndarray, list]], List[np.ndarray]]:
        """Merge/Aggregate tile-level prediction to image-level prediction.

        Args:
            results (list[list | tuple]): Testing tile results of the dataset.

        Returns:
            List[List]: Testing image results of the dataset.
        """
        assert len(results) == len(self.tiles)

        detection = False
        if isinstance(results[0], tuple):
            num_classes = len(results[0][0])
            dtype = results[0][0][0].dtype
        elif isinstance(results[0], list):
            detection = True
            num_classes = len(results[0])
            dtype = results[0][0].dtype
        else:
            raise RuntimeError("Unknown data type")

        merged_bbox_results: List[np.ndarray] = [np.empty((0, 5), dtype=dtype) for _ in range(self.num_images)]
        merged_mask_results: List[List] = [[] for _ in range(self.num_images)]
        merged_label_results: List[Union[List, np.ndarray]] = [np.array([]) for _ in range(self.num_images)]

        for result, tile in zip(results, self.tiles):
            tile_x1, tile_y1, _, _ = tile["tile_box"]
            img_idx = tile["dataset_idx"]
            img_h, img_w, _ = tile["original_shape_"]

            mask_result: List[List] = [[] for _ in range(num_classes)]
            if isinstance(result, tuple):
                bbox_result, mask_result = result
            else:
                bbox_result = result

            for cls_idx, cls_result in enumerate(zip(bbox_result, mask_result)):
                cls_bbox_result, cls_mask_result = cls_result
                _tmp_cls_bbox_result = np.zeros_like(cls_bbox_result)
                _tmp_cls_bbox_result[:, 0] = cls_bbox_result[:, 0] + tile_x1
                _tmp_cls_bbox_result[:, 1] = cls_bbox_result[:, 1] + tile_y1
                _tmp_cls_bbox_result[:, 2] = cls_bbox_result[:, 2] + tile_x1
                _tmp_cls_bbox_result[:, 3] = cls_bbox_result[:, 3] + tile_y1
                _tmp_cls_bbox_result[:, 4] = cls_bbox_result[:, 4]

                merged_bbox_results[img_idx] = np.concatenate((merged_bbox_results[img_idx], _tmp_cls_bbox_result))
                merged_label_results[img_idx] = np.concatenate(
                    [merged_label_results[img_idx], len(cls_bbox_result) * [cls_idx]]
                )

                for cls_mask_dict in cls_mask_result:
                    cls_mask_dict.update(dict(tile_box=tile["tile_box"], img_size=(img_h, img_w)))
                merged_mask_results[img_idx] += cls_mask_result

        # run NMS after aggregation suppressing duplicate boxes in
        # overlapping areas
        self.tile_nms(
            merged_bbox_results,
            merged_mask_results,
            merged_label_results,
            iou_threshold=self.iou_threshold,
            max_per_img=self.max_per_img,
            detection=detection,
        )

        assert len(merged_bbox_results) == len(merged_mask_results)
        if detection:
            return list(merged_bbox_results)
        return list(zip(merged_bbox_results, merged_mask_results))

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ann = {}
        if "gt_bboxes" in self.tiles[idx]:
            ann["bboxes"] = self.tiles[idx]["gt_bboxes"]
        if "gt_masks" in self.tiles[idx]:
            ann["masks"] = self.tiles[idx]["gt_masks"]
        if "gt_labels" in self.tiles[idx]:
            ann["labels"] = self.tiles[idx]["gt_labels"]
        return ann

    def merge_vectors(self, feature_vectors: List[np.ndarray]) -> np.ndarray:
        """Merge tile-level feature vectors to image-level feature vector.

        Args:
            feature_vectors (List[np.ndarray]): tile-level feature vectors.

        Returns:
            merged_vectors (List[np.ndarray]): Merged vectors for each image.
        """

        image_vectors: dict = {}
        for vector, tile in zip(feature_vectors, self.tiles):
            data_idx = tile.get("index", None) if "index" in tile else tile.get("dataset_idx", None)
            if data_idx in image_vectors:
                # tile vectors
                image_vectors[data_idx].append(vector)
            else:
                # whole image vector
                image_vectors[data_idx] = [vector]
        return [np.average(image, axis=0) for idx, image in image_vectors.items()]

    def merge_maps(self, saliency_maps: Union[List[List[np.ndarray]], List[np.ndarray]]) -> List:
        """Merge tile-level saliency maps to image-level saliency map.

        Args:
            saliency_maps (List[List[np.array] | np.ndarray]): tile-level saliency maps.
            Each map is a list of maps for each detected class or None if class wasn't detected.

        Returns:
            merged_maps (List[list | np.ndarray | None]): Merged saliency maps for each image.
        """

        dtype = None
        for map in saliency_maps:
            for cl_map in map:
                # find first class map which is not None
                if cl_map is not None and dtype is None:
                    dtype = cl_map.dtype
                    feat_h, feat_w = cl_map.shape
                    break
            if dtype is not None:
                break
        else:
            # if None for each class for each image
            return saliency_maps[: self.num_images]

        merged_maps = []
        ratios = {}
        num_classes = len(saliency_maps[0])

        for orig_image in self.cached_results:
            img_idx = orig_image["index"]
            image_h, image_w = orig_image["height"], orig_image["width"]
            ratios[img_idx] = np.array([feat_h / min(self.tile_size, image_h), feat_w / min(self.tile_size, image_w)])

            image_map_h = int(image_h * ratios[img_idx][0])
            image_map_w = int(image_w * ratios[img_idx][1])
            merged_maps.append([np.zeros((image_map_h, image_map_w)) for _ in range(num_classes)])

        for map, tile in zip(saliency_maps[self.num_images :], self.tiles[self.num_images :]):
            for class_idx in range(num_classes):
                if map[class_idx] is None:
                    continue
                cls_map = map[class_idx]
                img_idx = tile["dataset_idx"]
                x_1, y_1, x_2, y_2 = tile["tile_box"]
                y_1, x_1 = ((y_1, x_1) * ratios[img_idx]).astype(np.uint16)
                y_2, x_2 = ((y_2, x_2) * ratios[img_idx]).astype(np.uint16)

                map_h, map_w = cls_map.shape
                # resize feature map if it got from the tile which width and height is less the tile_size
                if (map_h > y_2 - y_1 > 0) and (map_w > x_2 - x_1 > 0):
                    cls_map = cv2.resize(cls_map, (x_2 - x_1, y_2 - y_1))
                # cut the rest of the feature map that went out of the image borders
                map_h, map_w = y_2 - y_1, x_2 - x_1

                for hi, wi in [(h_, w_) for h_ in range(map_h) for w_ in range(map_w)]:
                    map_pixel = cls_map[hi, wi]
                    # on tile overlap add 0.5 value of each tile
                    if merged_maps[img_idx][class_idx][y_1 + hi, x_1 + wi] != 0:
                        merged_maps[img_idx][class_idx][y_1 + hi, x_1 + wi] = 0.5 * (
                            map_pixel + merged_maps[img_idx][class_idx][y_1 + hi, x_1 + wi]
                        )
                    else:
                        merged_maps[img_idx][class_idx][y_1 + hi, x_1 + wi] = map_pixel

        norm_maps = []
        for merged_map, image_sal_map in zip(merged_maps, saliency_maps[: self.num_images]):
            for class_idx in range(num_classes):
                # don't have detections for this class on merged map
                if (merged_map[class_idx] == 0).all():
                    merged_map[class_idx] = None
                else:
                    image_map_cls = image_sal_map[class_idx]
                    # resize the feature map for whole image to add it to merged saliency maps
                    if image_map_cls is not None:
                        map_h, map_w = merged_map[class_idx].shape
                        image_map_cls = cv2.resize(image_map_cls, (map_w, map_h))
                        merged_map[class_idx] += (0.5 * image_map_cls).astype(dtype)
                    merged_map[class_idx] = non_linear_normalization(merged_map[class_idx])
            norm_maps.append(merged_map)

        return norm_maps
