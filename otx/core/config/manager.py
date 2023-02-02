"""Auto Configuration Manager ."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import Any, Dict, Optional

import mmcv
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import IDataset

from otx.core.data.manager.dataset_manager import DatasetManager


class ConfigManager:
    """Auto configuration manager that could set the proper configuration.

    Currently, it only supports the small amount of functions.
    * Data format detection
    * Task type detection
    * Write the data to the workspace
    * Write the data configuration to the workspace

    However, it will supports lots of things in the near future.
    * Automatic train type detection (Supervised, Self, Semi)
    * Automatic resource allocation (num_workers, HPO)

    """

    def __init__(self):
        # Currently, Datumaro.auto_split() can support below 3 tasks
        # Classification, Detection, Segmentation
        self.task_data_dict = {
            "CLASSIFICATION": ["imagenet"],
            "DETECTION": ["coco", "voc", "yolo"],
            "SEGMENTATION": ["common_semantic_segmentation", "voc", "cityscapes", "ade20k2017", "ade20k2020"],
            # "ACTION_CLASSIFICATION": ["multi-cvat"],
            # "ACTION_DETECTION": ["multi-cvat"],
            # "ANOMALY_CLASSIFICATION": ["mvtec"],
            # "ANOMALY_DETECTION": ["mvtec"],
            # "ANOMALY_SEGMENTATION": ["mvtec"],
            # "INSTANCE_SEGMENTATION": ["coco", "voc"],
        }
        self.data_format: str = ""
        self.task_type: str = ""
        self.splitted_dataset: Dict[str, IDataset] = {}

    def auto_task_detection(self, train_data_roots: str) -> str:
        """Detect task type automatically."""
        data_format = self._get_data_format(train_data_roots)
        return self._get_task_type(data_format)

    def _get_data_format(self, data_root: str) -> str:
        """Get dataset format."""
        self.data_format = DatasetManager.get_data_format(data_root)
        return self.data_format

    def _get_task_type(self, data_format: str) -> str:
        """Detect task type.

        For some datasets (i.e. COCO, VOC, MVTec), can't be fully automated.
        Because those datasets have several format at the same time.
        (i.e. for the COCO case, object detection and instance segmentation annotations coexist)
        In this case, the task_type will be selected to default value.

        For action tasks, currently action_classification is default.

        If Datumaro supports the Kinetics, AVA datasets, MVTec, _is_cvat_format(), _is_mvtec_format()
        functions will be deleted.
        """

        for task_key, data_value in self.task_data_dict.items():
            if data_format in data_value:
                self.task_type = task_key
                return task_key
        raise ValueError(f"Can't find proper task. we are not support {data_format} format, yet.")

    def auto_split_data(self, data_root: str, task: str):
        """Automatically Split train data --> train/val dataset."""
        self.data_format = DatasetManager.get_data_format(data_root)
        dataset = DatasetManager.import_dataset(data_root=data_root, data_format=self.data_format)
        self.splitted_dataset = DatasetManager.auto_split(
            task=task,
            dataset=DatasetManager.get_train_dataset(dataset),
            split_ratio=[("train", 0.8), ("val", 0.2)],
        )
        print("[*] Auto-split enabled.")

    def write_data_with_cfg(
        self,
        workspace_dir: str,
        train_data_roots: Optional[str] = None,
        val_data_roots: Optional[str] = None,
    ):
        """Save the splitted dataset and data.yaml to the workspace."""

        data_config = self._create_empty_data_cfg()
        if train_data_roots:
            data_config["data"]["train"]["data-roots"] = train_data_roots
        if val_data_roots:
            data_config["data"]["val"]["data-roots"] = val_data_roots

        default_data_folder_name = "splitted_dataset"

        self._save_data(workspace_dir, default_data_folder_name, data_config)

        self._export_data_cfg(data_config, os.path.join(workspace_dir, "data.yaml"))

    def _save_data(
        self,
        workspace_dir: str,
        default_data_folder_name: str,
        data_config: Dict[str, Dict[str, Dict[str, Any]]],
    ):
        """Save the data for the classification task.

        Args:
            workspace_dir (str): path of workspace
            default_data_folder_name (str): the name of splitted dataset folder
            data_config (dict): dictionary that has information about data path
        """
        for phase, dataset in self.splitted_dataset.items():
            dst_dir_path = os.path.join(workspace_dir, default_data_folder_name, phase)
            data_config["data"][phase]["data-roots"] = os.path.abspath(dst_dir_path)
            # Convert Datumaro class: DatasetFilter(IDataset) --> Dataset
            datum_dataset = Dataset.from_extractors(dataset)
            # Write the data
            # TODO: consider the way that reduces disk stroage
            # Currently, saving all images to the workspace.
            # It might needs quite large disk storage.
            DatasetManager.export_dataset(
                dataset=datum_dataset, output_dir=dst_dir_path, data_format=self.data_format, save_media=True
            )

    def _create_empty_data_cfg(
        self,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Create default dictionary to represent the dataset."""
        data_config: Dict[str, Dict[str, Any]] = {"data": {}}
        for subset in ["train", "val", "test", "unlabeled"]:
            data_subset = {"ann-files": None, "data-roots": None}
            data_config["data"][subset] = data_subset
        return data_config

    def _export_data_cfg(self, data_cfg: Dict[str, Dict[str, Dict[str, Any]]], output_path: str):
        """Export the data configuration file to output_path."""
        mmcv.dump(data_cfg, output_path)
        print(f"[*] Saving data configuration file to: {output_path}")
