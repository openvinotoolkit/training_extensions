# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tile Adaptor for OTX."""
from __future__ import annotations

import numpy as np
import logging as log
from otx.core.config import TileConfig


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
    stat["std"] = float(np.std(values))  # Normal scale std is better for understanding
    return stat


def compute_robust_dataset_statistics(dataset: DatasetEntity, ann_stat=False, max_samples=1000) -> Dict[str, Any]:
    """Computes robust statistics of image & annotation sizes.

    Args:
        dataset (DatasetEntity): Input dataset.
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

    max_image_samples = min(max_samples, len(dataset))
    image_indices = np.random.permutation(len(dataset))[:max_image_samples]

    image_sizes = []
    for i in image_indices:
        data = dataset[int(i)]
        image_sizes.append(np.sqrt(data.width * data.height))
    stat["image"] = compute_robust_scale_statistics(np.array(image_sizes))

    if ann_stat:
        stat["annotation"] = {}
        num_per_images: list[int] = []
        size_of_shapes: list[float] = []
        for i in image_indices:
            data = dataset[int(i)]
            annotations = data.get_annotations()
            num_per_images.append(len(annotations))

            if len(size_of_shapes) >= max_samples:
                continue

            image_area = data.width * data.height

            def scale_of(ann):
                return np.sqrt(image_area * ann.shape.get_area())

            size_of_shapes.extend(
                filter(lambda x: x >= 1, map(scale_of, annotations))
            )  # Filter out shapes smaller than 1 pixel as outlier

        stat["annotation"]["num_per_image"] = compute_robust_statistics(np.array(num_per_images))
        stat["annotation"]["size_of_shape"] = compute_robust_scale_statistics(np.array(size_of_shapes))

    return stat


def adapt_tile_config(tile_config: TileConfig, dataset: DatasetEntity):
    """Config tile parameters.

    Adapt based on annotation statistics.
    i.e. tile size, tile overlap, ratio and max objects per sample

    Args:
        tile_config (BaseTilingParameters): tiling parameters of the model
        dataset (DatasetEntity): training dataset
    """

    stat = compute_robust_dataset_statistics(dataset, ann_stat=True)
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

    # TODO [Eugene]: Need to discuss how we add parameter validators to dataclass config
    # if tile_overlap >= tile_config.tile_overlap["max_value"]:
    #     # Use the average object area if the tile overlap is too large to prevent 0 stride.
    #     tile_overlap = object_size / tile_size
    #     log.info(f"----> (too big) tile_overlap: {object_size} / {tile_size} = {tile_overlap}")

    # # validate parameters are in range
    # tile_size = max(
    #     tile_config.get_metadata("tile_size")["min_value"],
    #     min(tile_config.get_metadata("tile_size")["max_value"], tile_size),
    # )
    # tile_overlap = max(
    #     tile_config.get_metadata("tile_overlap")["min_value"],
    #     min(tile_config.get_metadata("tile_overlap")["max_value"], tile_overlap),
    # )
    # max_num_objects = max(
    #     tile_config.get_metadata("tile_max_number")["min_value"],
    #     min(tile_config.get_metadata("tile_max_number")["max_value"], max_num_objects),
    # )

    tile_config.tile_size = tile_size
    tile_config.tile_max_number = max_num_objects
    tile_config.tile_overlap = tile_overlap
