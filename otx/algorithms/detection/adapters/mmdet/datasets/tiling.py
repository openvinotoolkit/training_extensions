"""Tiling for detection and instance segmentation task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os.path as osp
import tempfile
import uuid
from itertools import product
from multiprocessing import Pool
from time import time
from typing import Callable, Dict, List, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.ops import nms
from mmdet.core import BitmapMasks, bbox2result
from tqdm import tqdm


def timeit(func) -> Callable:
    """Decorator to measure time of function execution.

    Args:
        func:

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
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests. Defaults to True.
        nproc (int, optional): Processes used for processing masks. Default: 4.
    """

    def __init__(
        self,
        dataset,
        pipeline,
        tmp_dir: tempfile.TemporaryDirectory,
        tile_size: int = 400,
        overlap: float = 0.2,
        min_area_ratio: float = 0.9,
        iou_threshold: float = 0.45,
        max_per_img: int = 1500,
        filter_empty_gt: bool = True,
        nproc: int = 4,
    ):
        self.min_area_ratio = min_area_ratio
        self.filter_empty_gt = filter_empty_gt
        self.iou_threshold = iou_threshold
        self.max_per_img = max_per_img
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.num_images = len(dataset)
        self.num_classes = len(dataset.CLASSES)
        self.CLASSES = dataset.CLASSES  # pylint: disable=invalid-name
        self.tmp_folder = tmp_dir.name
        self.nproc = nproc
        self.img2fp32 = False
        for p in pipeline:
            if p.type == "PhotoMetricDistortion":
                self.img2fp32 = True
                break

        self.dataset = dataset
        self.tiles = self.gen_tile_ann()
        self.cache_tiles()

    @timeit
    def cache_tiles(self):
        """Cache tiles to disk."""
        pbar = tqdm(total=len(self.tiles))
        pre_img_idx = None
        for i, tile in enumerate(self.tiles):
            tile["tile_path"] = osp.join(
                self.tmp_folder, "_".join([str(i), tile["uuid"], tile["ori_filename"], ".jpg"])
            )
            x_1, y_1, x_2, y_2 = tile["tile_box"]
            dataset_idx = tile["dataset_idx"]
            if dataset_idx != pre_img_idx:
                ori_img = self.dataset[dataset_idx]["img"]
                pre_img_idx = dataset_idx

            mmcv.imwrite(ori_img[y_1:y_2, x_1:x_2, :], tile["tile_path"])
            pbar.update(1)

    @timeit
    def gen_tile_ann(self) -> List[Dict]:
        """Generate tile information and tile annotation from dataset.

        Returns:
            List[Dict]: A list of tiles generated from the dataset. Each item comprises tile annotation and tile
                        coordinates relative to the original image.
        """
        tiles = []
        pbar = tqdm(total=len(self.dataset))

        for idx, result in enumerate(self.dataset):
            tiles.append(self.gen_single_img(result, dataset_idx=idx))

        for idx, result in enumerate(self.dataset):
            tiles.extend(self.gen_tiles_single_img(result, dataset_idx=idx))
            pbar.update(1)
        return tiles

    def gen_single_img(self, result: Dict, dataset_idx: int) -> Dict:
        """Add full-size image for inference or training.

        Args:
            result (Dict): the original image-level result (i.e. the original image annotation)
            dataset_idx (int): the image index this tile belongs to

        Returns:
            Dict: annotation with some other useful information for data pipeline.
        """
        result["tile_box"] = (0, 0, result["dataset_item"].width, result["dataset_item"].height)
        result["dataset_idx"] = dataset_idx
        result["original_shape_"] = result["img_shape"]
        result["uuid"] = str(uuid.uuid4())
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
        gt_bboxes = result.pop("gt_bboxes", np.zeros((0, 4), dtype=np.float32))
        gt_masks = result.pop("gt_masks", None)
        gt_bboxes_ignore = result.pop("gt_bboxes_ignore", np.zeros((0, 4), dtype=np.float32))
        gt_labels = result.pop("gt_labels", np.array([], dtype=np.int64))
        img_shape = result.pop("img_shape")
        height, width = img_shape[:2]
        _tile = self.prepare_result(result)

        num_patches_h = int((height - self.tile_size) / self.stride) + 1
        num_patches_w = int((width - self.tile_size) / self.stride) + 1
        for (_, _), (loc_i, loc_j) in zip(
            product(range(num_patches_h), range(num_patches_w)),
            product(
                range(0, height - self.tile_size + 1, self.stride),
                range(0, width - self.tile_size + 1, self.stride),
            ),
        ):
            x_1 = loc_j
            x_2 = loc_j + self.tile_size
            y_1 = loc_i
            y_2 = loc_i + self.tile_size
            tile = copy.deepcopy(_tile)
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
        overlap_ratio = self.tile_boxes_overlap(tile_box, gt_bboxes)
        match_idx = np.where((overlap_ratio[0] >= self.min_area_ratio))[0]

        if len(match_idx):
            tile_lables = gt_labels[match_idx][:]
            tile_bboxes = gt_bboxes[match_idx][:]
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
            tile_result["gt_masks"] = gt_masks[match_idx].crop(tile_box[0]) if gt_masks is not None else []
        else:
            tile_result.pop("bbox_fields")
            tile_result.pop("mask_fields")
            tile_result.pop("seg_fields")
            tile_result.pop("img_fields")
            tile_result["gt_bboxes"] = []
            tile_result["gt_labels"] = []
            tile_result["gt_masks"] = []

        if gt_masks is None:
            tile_result.pop("gt_masks")

    def tile_boxes_overlap(self, tile_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute overlapping ratio over boxes.

        Args:
            tile_box (np.ndarray): box in shape (1, 4).
            boxes (np.ndarray): boxes in shape (N, 4).

        Returns:
            np.ndarray: overlapping ratio over boxes
        """
        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        width_height = np.minimum(tile_box[:, None, 2:], boxes[:, 2:]) - np.maximum(tile_box[:, None, :2], boxes[:, :2])

        width_height = width_height.clip(min=0)  # [N,M,2]
        inter = width_height.prod(2)

        # handle empty boxes
        tile_box_ratio = np.where(inter > 0, inter / box_area, np.zeros(1, dtype=inter.dtype))
        return tile_box_ratio

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
                masks = [masks[keep_idx] for keep_idx in keep_indices]
                masks = self.process_masks(masks)
                mask_results[i] = [list(np.asarray(masks)[labels == i]) for i in range(self.num_classes)]

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
        if result.get("tile_path") and osp.isfile(result["tile_path"]):
            img = mmcv.imread(result["tile_path"])
            if self.img2fp32:
                img = img.astype(np.float32)
            result["img"] = img
            return result
        dataset_idx = result["dataset_idx"]
        x_1, y_1, x_2, y_2 = result["tile_box"]
        ori_img = self.dataset[dataset_idx]["img"]
        if self.img2fp32:
            ori_img = ori_img.astype(np.float32)
        result["img"] = ori_img[y_1:y_2, x_1:x_2, :]
        return result

    @staticmethod
    def readjust_tile_mask(tile_rle: Dict):
        """Shift tile-level mask to image-level mask.

        Args:
            tile_rle (Dict): _description_

        Returns:
            _type_: _description_
        """
        x1, y1, x2, y2 = tile_rle.pop("tile_box")
        height, width = tile_rle.pop("img_size")
        tile_mask = mask_util.decode(tile_rle)
        tile_mask = np.pad(tile_mask, ((y1, height - y2), (x1, width - x2)))
        return mask_util.encode(tile_mask)

    def process_masks(self, tile_masks: List[Dict]):
        """Decode Mask Result to Numpy mask, add paddings then encode masks again.

        Args:
            tile_masks (_type_): _description_

        Returns:
            _type_: _description_
        """
        with Pool(self.nproc) as pool:
            results = pool.map(Tile.readjust_tile_mask, tile_masks)
        return results

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
        merged_label_results: List[Union[List, np.ndarray]] = [[] for _ in range(self.num_images)]

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
