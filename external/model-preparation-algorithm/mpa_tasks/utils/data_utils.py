# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import fcntl
import os
import shutil
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


def clean_up_cache_dir(cache_dir: str):
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir)


def get_cached_image(results: Dict[str, Any], cache_dir: str, to_float32=False):
    def is_video_frame(media):
        return "VideoFrame" in repr(media)

    if is_video_frame(results["dataset_item"].media):
        subset = results["dataset_item"].subset
        index = results["index"]
        filename = os.path.join(cache_dir, f"{subset}-{index:06d}.npz")
        if os.path.exists(filename):
            # Might be slower than dict key checking, but persitent
            with open(filename, "rb") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    cached_img = np.load(f)
                    return cached_img['img']
                except Exception as e:
                    logger.warning(f"Skip loading cached {filename} \nError msg: {e}")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    img = results["dataset_item"].numpy  # this takes long for VideoFrame
    if to_float32:
        img = img.astype(np.float32)

    if is_video_frame(results["dataset_item"].media) and not os.path.exists(filename):
        with open(filename, "wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                np.savez_compressed(f, img=img)
            except Exception as e:
                logger.warning(f"Skip caching for {filename} \nError msg: {e}")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    return img
