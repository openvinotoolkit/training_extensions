# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tile Adaptor for OTX."""
from __future__ import annotations

import logging as log
from typing import Any

import numpy as np
from datumaro import Bbox, Dataset, DatasetSubset

from otx.core.config.data import TileConfig


def compute_robust_statistics(values: np.array) -> dict[str, float]:
    """Computes robust statistics of given samples.

    Args:
        values (np.array): Array of samples

    Returns:
        dict[str, float]: Robust avg, min, max values
    """
    stat: dict = {}
    if values.size == 0:
        return stat

    avg_value = np.mean(values)
    std_value = np.std(values)
    avg_3std_min_value = avg_value - 3 * std_value
    avg_3std_max_value = avg_value + 3 * std_value
    min_value = np.min(values)
    max_value = np.max(values)

    # Refine min/max to reduce outlier effect
    robust_min_value = max(min_value, avg_3std_min_value)
    robust_max_value = min(max_value, avg_3std_max_value)

    stat["avg"] = float(avg_value)
    stat["std"] = float(std_value)
    stat["min"] = float(min_value)
    stat["max"] = float(max_value)
    stat["robust_min"] = float(robust_min_value)
    stat["robust_max"] = float(robust_max_value)
    return stat


def compute_robust_scale_statistics(values: np.array) -> dict[str, float]:
    """Computes robust statistics of scale values.

    Average of 0.5x scale and 2x scale should be 1x

    Args:
        values (np.array): Array of positive scale values

    Returns:
        dict[str, float]: Robust avg, min, max values
    """
    # Compute stat in log scale & convert back to original scale
    if values.size == 0:
        return {}

    stat = compute_robust_statistics(np.log(values))
    stat = {k: float(np.exp(v)) for k, v in stat.items()}
    # Normal scale std is easier to understand
    stat["std"] = float(np.std(values))
    return stat


def compute_robust_dataset_statistics(
    dataset: DatasetSubset,
    ann_stat: bool = False,
    max_samples: int = 1000,
) -> dict[str, Any]:
    """Computes robust statistics of image & annotation sizes.

    Args:
        dataset (DatasetSubset): Input dataset.
        ann_stat (bool, optional): Whether to compute annotation size statistics. Defaults to False.
        max_samples (int, optional): Maximum number of dataset subsamples to analyze. Defaults to 1000.

    Returns:
        Dict[str, Any]: Robust avg, min, max values for images, and annotations optionally.
            ex) stat = {
                    "image": {"avg": ...},
                    "annotation": {
                       "num_per_image": {"avg": ...},
                       "size_of_shape": {"avg": ...},
                    }
                }
    """
    stat: dict = {}
    if len(dataset) == 0 or max_samples <= 0:
        return stat

    data_ids = [item.id for item in dataset]
    if len(dataset) > max_samples:
        log.warning(f"Too many samples. Only use first {max_samples} samples.")
        rng = np.random.default_rng()
        data_ids = rng.choice(data_ids, max_samples, replace=False)

    image_sizes = []
    for idx in data_ids:
        data = dataset.get(id=idx, subset=dataset.name)
        height, width = data.media.size
        image_sizes.append(np.sqrt(width * height))
    stat["image"] = compute_robust_scale_statistics(np.array(image_sizes))

    if ann_stat:
        stat["annotation"] = {}
        num_per_images: list[int] = []
        size_of_shapes: list[float] = []
        for idx in data_ids:
            data = dataset.get(id=idx, subset=dataset.name)
            annotations = [anno for anno in data.annotations if isinstance(anno, Bbox)]
            num_per_images.append(len(annotations))

            if len(size_of_shapes) >= max_samples:
                continue

            size_of_shapes.extend(
                filter(lambda x: x >= 1, [np.sqrt(anno.get_area()) for anno in annotations]),
            )

        stat["annotation"]["num_per_image"] = compute_robust_statistics(np.array(num_per_images))
        stat["annotation"]["size_of_shape"] = compute_robust_scale_statistics(np.array(size_of_shapes))

    return stat


def adapt_tile_config(tile_config: TileConfig, dataset: Dataset) -> None:
    """Config tile parameters.

    Adapt based on annotation statistics.
    i.e. tile size, tile overlap, ratio and max objects per sample

    Args:
        tile_config (TileConfig): tiling parameters of the model
        dataset (Dataset): Datumaro dataset including all subsets
    """
    if (train_dataset := dataset.subsets().get("train")) is not None:
        stat = compute_robust_dataset_statistics(train_dataset, ann_stat=True)
        max_num_objects = round(stat["annotation"]["num_per_image"]["max"])
        avg_size = stat["annotation"]["size_of_shape"]["avg"]
        min_size = stat["annotation"]["size_of_shape"]["robust_min"]
        max_size = stat["annotation"]["size_of_shape"]["robust_max"]
        log.info(f"----> [stat] scale avg: {avg_size}")
        log.info(f"----> [stat] scale min: {min_size}")
        log.info(f"----> [stat] scale max: {max_size}")

        object_size = avg_size

        log.info("[Adaptive tiling pararms]")
        object_tile_ratio = tile_config.object_tile_ratio
        tile_size = int(object_size / object_tile_ratio)
        tile_overlap = max_size / tile_size
        log.info(f"----> avg_object_size: {object_size}")
        log.info(f"----> max_object_size: {max_size}")
        log.info(f"----> object_tile_ratio: {object_tile_ratio}")
        log.info(f"----> tile_size: {object_size} / {object_tile_ratio} = {tile_size}")
        log.info(f"----> tile_overlap: {max_size} / {tile_size} = {tile_overlap}")

        if tile_overlap >= 0.9:
            # Use the average object area if the tile overlap is too large to prevent 0 stride.
            tile_overlap = object_size / tile_size
            log.info(f"----> (too big) tile_overlap: {object_size} / {tile_size} = {tile_overlap}")

        # TODO(Eugene): how to validate parameters? dataclass? pydantic?
        # https://github.com/openvinotoolkit/training_extensions/pull/2903
        tile_config.tile_size = (tile_size, tile_size)
        tile_config.max_num_instances = max_num_objects
        tile_config.overlap = tile_overlap
