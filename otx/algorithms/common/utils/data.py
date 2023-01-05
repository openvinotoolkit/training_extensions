"""Collections of Dataset utils for common OTX algorithms."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=invalid-name

import glob
import os
from typing import Optional

from otx.api.entities.annotation import NullAnnotationSceneEntity
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.subset import Subset
from otx.api.utils.argument_checks import (
    IMAGE_FILE_EXTENSIONS,
    DirectoryPathCheck,
    check_input_parameters_type,
)


@check_input_parameters_type({"file_list_path": DirectoryPathCheck})
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


@check_input_parameters_type({"data_root_dir": DirectoryPathCheck})
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
