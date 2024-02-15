# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Callback to set anchor statistics from train dataset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from datumaro.components.annotation import Bbox
from lightning import Callback

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator

    from otx.core.data.dataset.base import OTXDataset


logger = logging.getLogger()


class AdaptiveAnchorGenerator(Callback):
    """Adaptive anchor generating callback.

    Depending on Dataset's annotation statistiscs, this callback generate new anchors for SSD
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Execute this function at starting the train stage."""
        anchor_generator = pl_module.model.model.bbox_head.anchor_generator
        new_anchors = self._get_new_anchors(trainer.train_dataloader.dataset, anchor_generator)
        if new_anchors is not None:
            logger.warning("Anchor will be updated by Dataset's statistics")
            logger.warning(f"{anchor_generator.widths} -> {new_anchors[0]}")
            logger.warning(f"{anchor_generator.heights} -> {new_anchors[1]}")
            anchor_generator.widths = new_anchors[0]
            anchor_generator.heights = new_anchors[1]
            anchor_generator.gen_base_anchors()

    def _get_new_anchors(self, dataset: OTXDataset, anchor_generator: AnchorGenerator) -> tuple | None:
        """Get new anchors for SSD from OTXDataset."""
        from mmdet.datasets.transforms import Resize

        target_wh = None
        if isinstance(dataset.transforms, list):
            for transform in dataset.transforms:
                if isinstance(transform, Resize):
                    target_wh = transform.scale
        if target_wh is None:
            target_wh = (864, 864)
            msg = f"Cannot get target_wh from the dataset. Assign it with the default value: {target_wh}"
            logger.warning(msg)
        group_as = [len(width) for width in anchor_generator.widths]
        wh_stats = self._get_sizes_from_dataset_entity(dataset, list(target_wh))

        if len(wh_stats) < sum(group_as):
            logger.warning(
                f"There are not enough objects to cluster: {len(wh_stats)} were detected, while it should be "
                f"at least {sum(group_as)}. Anchor box clustering was skipped.",
            )
            return None

        return self._get_anchor_boxes(wh_stats, group_as)

    @staticmethod
    def _get_sizes_from_dataset_entity(dataset: OTXDataset, target_wh: list[int]) -> list[tuple[int, int]]:
        """Function to get width and height size of items in OTXDataset.

        Args:
            dataset(OTXDataset): OTXDataset in which to get statistics
            target_wh(list[int]): target width and height of the dataset
        Return
            list[tuple[int, int]]: tuples with width and height of each instance
        """
        wh_stats: list[tuple[int, int]] = []
        for item in dataset.dm_subset:
            for ann in item.annotations:
                if isinstance(ann, Bbox):
                    x1, y1, x2, y2 = ann.points
                    x1 = x1 / item.media.size[1] * target_wh[0]
                    y1 = y1 / item.media.size[0] * target_wh[1]
                    x2 = x2 / item.media.size[1] * target_wh[0]
                    y2 = y2 / item.media.size[0] * target_wh[1]
                    wh_stats.append((x2 - x1, y2 - y1))
        return wh_stats

    @staticmethod
    def _get_anchor_boxes(wh_stats: list[tuple[int, int]], group_as: list[int]) -> tuple:
        """Get new anchor box widths & heights using KMeans."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(init="k-means++", n_clusters=sum(group_as), random_state=0).fit(wh_stats)
        centers = kmeans.cluster_centers_

        areas = np.sqrt(np.prod(centers, axis=1))
        idx = np.argsort(areas)

        widths = centers[idx, 0]
        heights = centers[idx, 1]

        group_as = np.cumsum(group_as[:-1])
        widths, heights = np.split(widths, group_as), np.split(heights, group_as)
        widths = [width.tolist() for width in widths]
        heights = [height.tolist() for height in heights]
        return widths, heights
