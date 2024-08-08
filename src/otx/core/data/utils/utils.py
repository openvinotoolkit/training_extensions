# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for the data module."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from datumaro.components.annotation import _Shape

if TYPE_CHECKING:
    from datumaro import Dataset, DatasetSubset

    from otx.core.config.data import TileConfig


logger = logging.getLogger(__name__)


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


def compute_robust_dataset_statistics(dataset: DatasetSubset, max_samples: int = 1000) -> dict[str, Any]:
    """Computes robust statistics of image & annotation sizes.

    Args:
        dataset (DatasetSubset): Input dataset.
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
    max_image_samples = min(max_samples, len(dataset))
    rng = np.random.default_rng(42)
    data_ids = rng.choice(data_ids, max_image_samples, replace=False)[:max_image_samples]

    image_sizes = []
    for idx in data_ids:
        data = dataset.get(id=idx, subset=dataset.name)
        height, width = data.media.size
        image_sizes.append(np.sqrt(width * height))
    stat["image"] = compute_robust_scale_statistics(np.array(image_sizes))

    stat["annotation"] = {}
    num_per_images: list[int] = []
    size_of_shapes: dict[str, list] = defaultdict(list)
    for idx in data_ids:
        data = dataset.get(id=idx, subset=dataset.name)
        annotations: dict[str, list] = defaultdict(list)
        for ann in data.annotations:
            annotations[ann.__class__.__name__].append(ann)

        num_per_images.append(max(len(val) for val in annotations.values()) if annotations else 0)

        if size_of_shapes and max(len(val) for val in size_of_shapes.values()) >= max_samples:
            continue

        for ann_type, anns in annotations.items():
            size_of_shapes[ann_type].extend(
                np.sqrt(area) for val in anns if isinstance(val, _Shape) and (area := val.get_area()) >= 1
            )

    stat["annotation"]["num_per_image"] = compute_robust_statistics(np.array(num_per_images))
    if "Polygon" in size_of_shapes:
        stat["annotation"]["size_of_shape"] = compute_robust_scale_statistics(np.array(size_of_shapes["Polygon"]))
    else:
        max_ann_type = None
        max_num_ann = 0
        for ann_type, anns in size_of_shapes.items():
            if max_num_ann < len(anns):
                max_ann_type = ann_type
                max_num_ann = len(anns)
        if max_ann_type is not None:
            stat["annotation"]["size_of_shape"] = compute_robust_scale_statistics(
                np.array(size_of_shapes[max_ann_type]),
            )

    return stat


def adapt_input_size_to_dataset(
    dataset: Dataset,
    base_input_size: int | tuple[int, int] | None = None,
    downscale_only: bool = True,
    input_size_multiplier: int | None = None,
) -> tuple[int, int] | None:
    """Compute appropriate model input size w.r.t. dataset statistics.

    Args:
        dataset (Dataset): Datumaro dataset including all subsets.
        base_input_size (int | tuple[int, int] | None, optional): Base input size of the model. Defaults to None.
        downscale_only (bool, optional) : Whether to allow only smaller size than default setting. Defaults to True.
        input_size_multiplier (int | None, optional):
            Multiplier for input size. If it's set, return the input size which can be divisible by the value.
            Defaults to None.

    Returns:
        tuple[int, int] | None: Recommended input size based on dataset statistics.
    """
    min_recognizable_object_size = 32  # Minimum object size recognizable by NNs: typically 16 ~ 32
    # meaning NxN input pixels being downscaled to 1x1 on feature map
    min_detection_input_size = 256  # Minimum input size for object detection

    if downscale_only and base_input_size is None:
        msg = "If downscale_only is set to True, base_input_size should be set but got None."
        raise ValueError(msg)

    if isinstance(base_input_size, int):
        base_input_size = (base_input_size, base_input_size)

    train_dataset = dataset.subsets().get("train")
    if train_dataset is None:
        return None

    logger.info("Adapting model input size based on dataset stat")
    stat = compute_robust_dataset_statistics(train_dataset)
    max_image_size = stat["image"]["robust_max"]
    min_object_size = None
    if stat["annotation"]:
        # Refine using annotation shape size stat
        # Fit to typical small object size (conservative)
        # -> "avg" size might be preferrable for efficiency
        min_object_size = stat["annotation"].get("size_of_shape", {}).get("robust_min", None)

    logger.info(f"-> Current base input size: {base_input_size}")

    if max_image_size <= 0:
        return base_input_size

    image_size = max_image_size
    logger.info(f"-> Based on typical large image size: {image_size}")

    # Refine using annotation shape size stat
    if min_object_size is not None and min_object_size > 0:
        image_size = round(image_size * min_recognizable_object_size / min_object_size)
        logger.info(f"-> Based on typical small object size {min_object_size}: {image_size}")
        if image_size > max_image_size:
            image_size = max_image_size
            logger.info(f"-> Restrict to max image size: {image_size}")
        if image_size < min_detection_input_size:
            image_size = min_detection_input_size
            logger.info(f"-> Based on minimum object detection input size: {image_size}")

    if input_size_multiplier is not None and image_size % input_size_multiplier != 0:
        image_size = (image_size // input_size_multiplier + 1) * input_size_multiplier

    input_size = (round(image_size), round(image_size))

    if downscale_only:

        def area(x: tuple[int, int]) -> int:
            return x[0] * x[1]

        if base_input_size and area(input_size) >= area(base_input_size):
            logger.info(f"-> Downscale only: {input_size} -> {base_input_size}")
            return base_input_size

    # Closest preset
    logger.info(f"-> Closest preset: {input_size}")
    return input_size


def adapt_tile_config(tile_config: TileConfig, dataset: Dataset) -> None:
    """Config tile parameters.

    Adapt based on annotation statistics.
    i.e. tile size, tile overlap, ratio and max objects per sample

    Args:
        tile_config (TileConfig): tiling parameters of the model
        dataset (Dataset): Datumaro dataset including all subsets
    """
    if (train_dataset := dataset.subsets().get("train")) is not None:
        stat = compute_robust_dataset_statistics(train_dataset)
        max_num_objects = round(stat["annotation"]["num_per_image"]["max"])
        avg_size = stat["annotation"]["size_of_shape"]["avg"]
        min_size = stat["annotation"]["size_of_shape"]["robust_min"]
        max_size = stat["annotation"]["size_of_shape"]["robust_max"]
        logger.info(f"----> [stat] scale avg: {avg_size}")
        logger.info(f"----> [stat] scale min: {min_size}")
        logger.info(f"----> [stat] scale max: {max_size}")

        logger.info("[Adaptive tiling pararms]")
        object_tile_ratio = tile_config.object_tile_ratio
        tile_size = int(avg_size / object_tile_ratio)
        tile_overlap = max_size / tile_size
        logger.info(f"----> avg_object_size: {avg_size}")
        logger.info(f"----> max_object_size: {max_size}")
        logger.info(f"----> object_tile_ratio: {object_tile_ratio}")
        logger.info(f"----> tile_size: {avg_size} / {object_tile_ratio} = {tile_size}")
        logger.info(f"----> tile_overlap: {max_size} / {tile_size} = {tile_overlap}")

        if tile_overlap >= 0.9:
            # Use the average object area if the tile overlap is too large to prevent 0 stride.
            tile_overlap = min(avg_size / tile_size, 0.9)
            logger.info(f"----> (too big) tile_overlap: {avg_size} / {tile_size} = min[{tile_overlap}, 0.9]")

        # TODO(Eugene): how to validate lower/upper_bound? dataclass? pydantic?
        # https://github.com/openvinotoolkit/training_extensions/pull/2903
        tile_config.tile_size = (tile_size, tile_size)
        tile_config.max_num_instances = max_num_objects
        tile_config.overlap = tile_overlap
