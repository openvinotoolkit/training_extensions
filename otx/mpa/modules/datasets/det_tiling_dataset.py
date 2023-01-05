# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import tempfile
from typing import Dict

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

from .pipelines.tiling import Tile


@DATASETS.register_module()
class ImageTilingDataset(object):
    """A wrapper of tiling dataset.

    Suitable for training small object dataset. This wrapper composed of `Tile`
    that crops an image into tiles and merges tile-level predictions to
    image-level prediction for evaluation.
    Args:
        dataset (CustomDataset): The dataset to be tiled.
        pipeline (List): Sequence of transform object or
            config dict to be composed.
        tile_size (int): the length of side of each tile
        min_area_ratio (float, optional): The minimum overlap area ratio
            between a tiled image and its annotations. Ground-truth box is
            discarded if the overlap area is less than this value.
            Defaults to 0.8.
        overlap_ratio (float, optional): ratio of each tile to overlap with
            each of the tiles in its 4-neighborhood. Defaults to 0.2.
        iou_threshold (float, optional): IoU threshold to be used to suppress
            boxes in tiles' overlap areas. Defaults to 0.45.
        max_per_img (int, optional): if there are more than max_per_img bboxes
            after NMS, only top max_per_img will be kept. Defaults to 200.
    """

    def __init__(
        self,
        dataset,
        pipeline: list,
        tile_size: int,
        min_area_ratio=0.8,
        overlap_ratio=0.2,
        iou_threshold=0.45,
        max_per_img=200,
        filter_empty_gt=True,
        test_mode=False,
    ):

        self.CLASSES = dataset.CLASSES
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset = dataset
        self.tile_dataset = Tile(
            self.dataset,
            pipeline,
            tmp_dir=self.tmp_dir,
            tile_size=tile_size,
            overlap=overlap_ratio,
            min_area_ratio=min_area_ratio,
            iou_threshold=iou_threshold,
            max_per_img=max_per_img,
            filter_empty_gt=False if dataset.test_mode else filter_empty_gt,
        )
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.num_samples = len(dataset)  # number of original samples
        self.merged_results = None

    def __len__(self) -> int:
        return len(self.tile_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get training/test tile.
        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
            True).
        """
        return self.pipeline(self.tile_dataset[idx])

    def evaluate(self, results, **kwargs) -> Dict[str, float]:
        """Evaluation on Tile dataset.

        Args:
            results (list[list | tuple]): Testing results of the dataset.

        Returns:
            dict[str, float]: evaluation metric.
        """
        self.merged_results = self.tile_dataset.merge(results)
        return self.dataset.evaluate(self.merged_results, **kwargs)

    def merge(self, results):
        self.merged_results = self.tile_dataset.merge(results)
        return self.merged_results

    def __del__(self):
        if getattr(self, "tmp_dir", False):
            self.tmp_dir.cleanup()
