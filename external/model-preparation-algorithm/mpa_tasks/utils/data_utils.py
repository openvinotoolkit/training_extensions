# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import fcntl
import os
import cv2
from typing import Any, Dict

import numpy as np
from mpa.utils.logger import get_logger

logger = get_logger()


def get_cls_img_indices(labels, dataset):
    img_indices = {label.name: list() for label in labels}
    for i, item in enumerate(dataset):
        item_labels = item.annotation_scene.get_labels()
        for i_l in item_labels:
            if i_l in labels:
                img_indices[i_l.name].append(i)

    return img_indices


def get_old_new_img_indices(labels, new_classes, dataset):
    ids_old, ids_new = [], []
    _dataset_label_schema_map = {label.name: label for label in labels}
    new_classes = [_dataset_label_schema_map[new_class] for new_class in new_classes]
    for i, item in enumerate(dataset):
        if item.annotation_scene.contains_any(new_classes):
            ids_new.append(i)
        else:
            ids_old.append(i)
    return {"old": ids_old, "new": ids_new}


def get_image(results: Dict[str, Any], cache_dir: str, to_float32=False):
    def is_video_frame(media):
        return "VideoFrame" in repr(media)

    def is_training_subset(subset):
        return subset.name in ["TRAINING", "VALIDATION"]

    def load_image_from_cache(filename: str, to_float32=False):
        try:
            f = open(filename, "rb")
            fcntl.flock(f, fcntl.LOCK_SH)
            cached_img = np.asarray(bytearray(f.read()))
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
            cached_img = cv2.imdecode(cached_img, cv2.IMREAD_COLOR)
            if to_float32:
                cached_img = cached_img.astype(np.float32)
            return cached_img
        except Exception as e:
            logger.warning(f"Skip loading cached {filename} \nError msg: {e}")
            try:  # to handle the case that f is not defined: failed open(filename, "rb")
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            except Exception:
                return None

    def save_image_to_cache(img: np.array, filename: str):
        try:
            f = open(filename, "wb")
            fcntl.flock(f, fcntl.LOCK_EX)
            _, binary_img = cv2.imencode('.png', img)  # imencode returns (compress_flag, binary_img)
            f.write(binary_img)
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        except Exception as e:
            logger.warning(f"Skip caching for {filename} \nError msg: {e}")
            try:  # to handle the case that f is not defined: failed open(filename, "wb")
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            except Exception:
                return None

    subset = results["dataset_item"].subset
    if is_training_subset(subset) and is_video_frame(results["dataset_item"].media):
        index = results["index"]
        filename = os.path.join(cache_dir, f"{subset}-{index:06d}.png")
        if os.path.exists(filename):
            loaded_img = load_image_from_cache(filename, to_float32=to_float32)
            if loaded_img is not None:
                return loaded_img

    img = results["dataset_item"].numpy  # this takes long for VideoFrame
    if to_float32:
        img = img.astype(np.float32)

    if is_training_subset(subset) and is_video_frame(results["dataset_item"].media) and not os.path.exists(filename):
        save_image_to_cache(img, filename)

    return img
