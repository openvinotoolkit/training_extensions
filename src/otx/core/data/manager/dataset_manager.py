"""Datumaro Helper."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name
import os
from typing import List, Optional, Tuple, Union

import datumaro
from datumaro.components.dataset import Dataset, DatasetSubset
from datumaro.components.dataset_base import DatasetItem
from datumaro.plugins.splitter import Split


class DatasetManager:
    """The aim of DatasetManager is support datumaro functions at easy use.

    All kind of functions implemented in Datumaro are supported by this Manager.
    Since DatasetManager just wraps Datumaro's function,
    All methods are implemented as static method.
    """

    @staticmethod
    def get_train_dataset(dataset: Dataset) -> DatasetSubset:
        """Returns train dataset."""
        subsets = dataset.subsets()
        train_dataset = subsets.get("train", None)

        if train_dataset is not None:
            return train_dataset

        for k, v in subsets.items():
            if "train" in k or "default" in k:
                return v
        raise ValueError("Can't find training data.")

    @staticmethod
    def get_val_dataset(dataset: Dataset) -> Union[DatasetSubset, None]:
        """Returns validation dataset."""
        subsets = dataset.subsets()
        val_dataset = subsets.get("val", None)

        if val_dataset is not None:
            return val_dataset

        for k, v in subsets.items():
            if "val" in k:
                return v
        return None

    @staticmethod
    def get_data_format(data_root: str) -> str:
        """Find the format of dataset."""
        data_root = os.path.abspath(data_root)

        data_format: str = ""

        # TODO #
        # Currently, below `if/else` statements is mandatory
        # because Datumaro can't detect the multi-cvat and mvtec.
        # After, the upgrade of Datumaro, below codes will be changed.
        if DatasetManager.is_cvat_format(data_root):
            data_format = "multi-cvat"
        elif DatasetManager.is_mvtec_format(data_root):
            data_format = "mvtec"
        else:
            data_formats = datumaro.Environment().detect_dataset(data_root)
            # TODO: how to avoid hard-coded part
            data_format = data_formats[0] if "imagenet" not in data_formats else "imagenet"
        print(f"[*] Detected dataset format: {data_format}")
        return data_format

    @staticmethod
    def get_image_path(data_item: DatasetItem) -> Optional[str]:
        """Returns the path of image."""
        if hasattr(data_item.media, "path"):
            return data_item.media.path
        return None

    @staticmethod
    def export_dataset(dataset: Dataset, output_dir: str, data_format: str, save_media=True):
        """Export the Datumaro Dataset."""
        return dataset.export(output_dir, data_format, save_media=save_media)

    @staticmethod
    def import_dataset(data_root: str, data_format: str, subset: Optional[str] = None) -> dict:
        """Import dataset."""
        return Dataset.import_from(data_root, format=data_format, subset=subset)

    @staticmethod
    def auto_split(task: str, dataset: Dataset, split_ratio: List[Tuple[str, float]]) -> dict:
        """Automatically split the dataset: train --> train/val."""
        splitter = Split(dataset, task.lower(), split_ratio)
        return splitter.subsets()

    @staticmethod
    def is_cvat_format(path: str) -> bool:
        """Detect whether data path is CVAT format or not.

        Currently, we used multi-video CVAT format for Action tasks.

        This function can detect the multi-video CVAT format.

        Multi-video CVAT format
        root
        |--video_0
            |--images
                |--frame0001.png
            |--annotations.xml
        |--video_1
        |--video_2

        will be deprecated soon.
        """

        cvat_format = sorted(["images", "annotations.xml"])
        for sub_folder in os.listdir(path):
            # video_0, video_1, ...
            sub_folder_path = os.path.join(path, sub_folder)
            # files must be same with cvat_format
            if os.path.isdir(sub_folder_path):
                files = sorted(os.listdir(sub_folder_path))
                if files != cvat_format:
                    return False
        return True

    @staticmethod
    def is_mvtec_format(path: str) -> bool:
        """Detect whether data path is MVTec format or not.

        Check the first-level architecture folder, to know whether the dataset is MVTec or not.

        MVTec default structure like as below:
        root
        |--ground_truth
        |--train
        |--test

        will be deprecated soon.
        """

        mvtec_format = sorted(["ground_truth", "train", "test"])
        folder_list = []
        for sub_folder in os.listdir(path):
            sub_folder_path = os.path.join(path, sub_folder)
            # only use the folder name.
            if os.path.isdir(sub_folder_path):
                folder_list.append(sub_folder)
        return sorted(folder_list) == mvtec_format
