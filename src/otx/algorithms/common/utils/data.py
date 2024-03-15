"""Collections of Dataset utils for common OTX algorithms."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

import glob
import os
import random
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from otx.api.entities.annotation import NullAnnotationSceneEntity
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.subset import Subset
from otx.api.utils.argument_checks import IMAGE_FILE_EXTENSIONS
from otx.utils.logger import get_logger

logger = get_logger()


def get_unlabeled_filename(base_root: str, file_list_path: str):
    """This method checks and gets image file paths, which are listed in file_list_path.

    The content of file_list_path is expected to specify relative paths of each image file to base_root line by line.
    It returns the list of image filenames only which will compose unlabeled dataset.

    Args:
        base_root (str): path of base root dir where unlabeled images are.
        file_list_path (str) : path of file which contains relative paths of unlabeled data to base_root.

    Returns:
        List[str]: a list of existing image file paths which will be unlabeled data items.
    """

    def is_valid(file_path):
        return file_path.lower().endswith(tuple(IMAGE_FILE_EXTENSIONS))

    with open(file_list_path, "r", encoding="UTF-8") as f:
        file_names = f.read().splitlines()
    unlabeled_files = []
    for fn in file_names:
        file_path = os.path.join(base_root, fn.strip())
        if is_valid(file_path) and os.path.isfile(file_path):
            unlabeled_files.append(file_path)
    return unlabeled_files


def load_unlabeled_dataset_items(
    data_root_dir: str,
    file_list_path: Optional[str] = None,
):
    """This method loads unlabeled dataset items from images in data_root_dir.

    Args:
        data_root_dir (str): path of base root directory where unlabeled images are.
        file_list_path (str) : path of a file which contains relative paths of unlabeled data to base_root.
        subset (Subset) : Entity subset category
    Returns:
        List[DatasetItemEntity]: a list of unlabeled dataset item entity.
    """
    if file_list_path is not None:
        data_list = get_unlabeled_filename(data_root_dir, file_list_path)

    else:
        data_list = []

        for ext in IMAGE_FILE_EXTENSIONS:
            data_list.extend(glob.glob(f"{data_root_dir}/**/*{ext}", recursive=True))

    dataset_items = []

    for filename in data_list:
        dataset_item = DatasetItemEntity(
            media=Image(file_path=filename),
            annotation_scene=NullAnnotationSceneEntity(),
            subset=Subset.UNLABELED,
        )
        dataset_items.append(dataset_item)
    return dataset_items


def get_dataset(dataset: DatasetEntity, subset: Subset):
    """Get dataset from datasetentity."""
    data = dataset.get_subset(subset)
    return data if len(data) > 0 else None


def get_cls_img_indices(labels, dataset):
    """Function for getting image indices per class.

    Args:
        labels (List[LabelEntity]): List of labels
        dataset(DatasetEntity): dataset entity
    """
    img_indices = {label.name: [] for label in labels}
    for i, item in enumerate(dataset):
        item_labels = item.annotation_scene.get_labels()
        for i_l in item_labels:
            if i_l in labels:
                img_indices[i_l.name].append(i)

    return img_indices


def get_old_new_img_indices(labels, new_classes, dataset):
    """Function for getting old & new indices of dataset.

    Args:
        labels (List[LabelEntity]): List of labels
        new_classes(List[str]): List of new classes
        dataset(DatasetEntity): dataset entity
    """
    ids_old, ids_new = [], []
    _dataset_label_schema_map = {label.name: label for label in labels}
    new_classes = [_dataset_label_schema_map[new_class] for new_class in new_classes]
    for i, item in enumerate(dataset):
        if item.annotation_scene.contains_any(new_classes):
            ids_new.append(i)
        else:
            ids_old.append(i)
    return {"old": ids_old, "new": ids_new}


def get_image(results: Dict[str, Any], cache_dir: str, to_float32=False) -> np.ndarray:
    """Load an image and cache it if it's a training video frame.

    Args:
        results (Dict[str, Any]): A dictionary that contains information about the dataset item.
        cache_dir (str): A directory path where the cached images will be stored.
        to_float32 (bool, optional): A flag indicating whether to convert the image to float32. Defaults to False.

    Returns:
        np.ndarray: The loaded image.
    """

    def is_training_video_frame(subset, media) -> bool:
        return subset.name in ["TRAINING", "VALIDATION"] and "VideoFrame" in repr(media)

    def load_image_from_cache(filename: str, to_float32=False) -> Union[np.ndarray, None]:
        try:
            cached_img = cv2.imread(filename)
            if to_float32:
                cached_img = cached_img.astype(np.float32)
            return cached_img
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Skip loading cached {filename} \nError msg: {e}")
            return None

    def save_image_to_cache(img: np.array, filename: str):
        tmp_filename = filename.replace(".png", "-tmp.png")
        if os.path.exists(filename) or os.path.exists(tmp_filename):  # if image is cached or caching
            return
        try:
            cv2.imwrite(tmp_filename, img=img)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Skip caching for {filename} \nError msg: {e}")
            return

        if os.path.exists(tmp_filename) and not os.path.exists(filename):
            try:
                os.replace(tmp_filename, filename)
            except Exception as e:  # pylint: disable=broad-except
                os.remove(tmp_filename)
                logger.warning(f"Failed to rename {tmp_filename} -> {filename} \nError msg: {e}")

    subset = results["dataset_item"].subset
    media = results["dataset_item"].media
    if is_training_video_frame(subset, media):
        index = results["index"]
        filename = os.path.join(cache_dir, f"{subset}-{index:06d}.png")
        if os.path.exists(filename):
            loaded_img = load_image_from_cache(filename, to_float32=to_float32)
            if loaded_img is not None:
                return loaded_img

    img = results["dataset_item"].numpy  # this takes long for VideoFrame
    if to_float32:
        img = img.astype(np.float32)

    if is_training_video_frame(subset, media):
        save_image_to_cache(img, filename)

    return img


class OTXOpenVinoDataLoader:
    """DataLoader implementation for ClassificationOpenVINOTask."""

    def __init__(self, dataset: DatasetEntity, inferencer: Any, shuffle: bool = True):
        super().__init__()
        self.dataset = dataset
        self.inferencer = inferencer
        self.shuffler = None
        if shuffle:
            self.shuffler = list(range(len(dataset)))
            random.shuffle(self.shuffler)

    def __getitem__(self, index: int):
        """Get item from dataset."""
        if self.shuffler is not None:
            index = self.shuffler[index]

        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene

        resized_image = self.inferencer.model.resize(image, (self.inferencer.model.w, self.inferencer.model.h))
        resized_image = self.inferencer.model.input_transform(resized_image)
        resized_image = self.inferencer.model._change_layout(resized_image)

        return resized_image, annotation

    def __len__(self):
        """Get length of dataset."""

        return len(self.dataset)


def compute_robust_statistics(values: np.array) -> Dict[str, float]:
    """Computes robust statistics of given samples.

    Args:
        values (np.array): Array of samples

    Returns:
        Dict[str, float]: Robust avg, min, max values
    """
    stat: Dict = {}
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


def compute_robust_scale_statistics(values: np.array) -> Dict[str, float]:
    """Computes robust statistics of scale values.

    Average of 0.5x scale and 2x scale should be 1x

    Args:
        values (np.array): Array of positive scale values

    Returns:
        Dict[str, float]: Robust avg, min, max values
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
    stat: Dict = {}
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
        num_per_images: List[int] = []
        size_of_shapes: List[float] = []
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
