"""Util functions for auto configuration."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Optional, Union

TASK_TYPE_TO_SUPPORTED_FORMAT = {
    "CLASSIFICATION": ["imagenet", "datumaro"],
    "DETECTION": ["coco", "voc", "yolo"],
    "SEGMENTATION": ["cityscapes", "common_semantic_segmentation", "voc", "ade20k2017", "ade20k2020"],
    "ACTION_CLASSIFICATION": ["multi-cvat"],
    "ACTION_DETECTION": ["multi-cvat"],
    "ANOMALY_CLASSIFICATION": ["mvtec"],
    "ANOMALY_DETECTION": ["mvtec"],
    "ANOMALY_SEGMENTATION": ["mvtec"],
    "INSTANCE_SEGMENTATION": ["coco", "voc"],
    "ROTATED_DETECTION": ["coco", "voc"],
}


def configure_task_type(
    data_roots: Optional[str] = None,
    data_format: Optional[str] = None,
) -> tuple:
    if data_format is None and data_roots is not None:
        try:
            from otx.v2.adapters.datumaro.manager.dataset_manager import DatasetManager
        except ImportError as err:
            msg = "Need datumaro to automatically detect the task type."
            raise ImportError(msg) from err
        data_format = DatasetManager.get_data_format(data_roots)
    for task_key, data_value in TASK_TYPE_TO_SUPPORTED_FORMAT.items():
        if data_format in data_value:
            return task_key, data_format
    msg = f"Can't find proper task. we are not support {data_format} format, yet."
    raise ValueError(msg)


def configure_train_type(train_data_roots: Optional[str], unlabeled_data_roots: Optional[str]) -> Optional[str]:
    """Auto train type detection.

    If train_data_roots contains only set of images -> Self-SL
    If unlabeled-data-roots were passed -> use Semi-SL
    If unlabeled_images presented in dataset structure and it is sufficient to start Semi-SL -> Semi-SL
    Overwise set Incremental training type.
    """

    def _count_imgs_in_dir(dir_path: Union[str, Path]) -> int:
        """Count number of images in directory recursively."""
        valid_suff = ["jpg", "png", "jpeg", "gif"]
        num_valid_imgs = 0
        for file_path in Path(dir_path).rglob("*"):
            if file_path.suffix[1:].lower() in valid_suff and file_path.is_file():
                num_valid_imgs += 1

        return num_valid_imgs

    def _check_semisl_requirements(unlabeled_dir: Optional[Union[str, Path]]) -> Union[bool, str, Path]:
        """Check if quantity of unlabeled images is sufficient for Semi-SL learning."""
        if unlabeled_dir is None:
            return False

        if not Path(unlabeled_dir).is_dir() or not Path(unlabeled_dir).iterdir():
            msg = "unlabeled-data-roots isn't a directory, it doesn't exist or it is empty. Please, check command line and directory path."
            raise ValueError(
                msg,
            )

        all_unlabeled_images = _count_imgs_in_dir(unlabeled_dir)
        # check if number of unlabeled images is more than relative thershold
        if all_unlabeled_images > 1:
            return unlabeled_dir

        logging.warning(
            "WARNING: There are none or too litle images to start Semi-SL training. "
            "It should be more than relative threshold (at least 7% of labeled images) "
            "Start Supervised training instead.",
        )
        return False

    if train_data_roots is None or not Path(train_data_roots).is_dir() or not Path(train_data_roots).iterdir():
        return None

    if _count_imgs_in_dir(train_data_roots):
        # If train folder with images only was passed to args
        # Then we start self-supervised training
        print("[*] Selfsupervised training type detected")
        return "Selfsupervised"

    # if user explicitly passed unlabeled images folder
    valid_unlabeled_path = _check_semisl_requirements(unlabeled_data_roots)
    if valid_unlabeled_path:
        print(f"[*] Semisupervised training type detected with unlabeled data: {valid_unlabeled_path}")
        return "Semisupervised"
    return "Incremental"
