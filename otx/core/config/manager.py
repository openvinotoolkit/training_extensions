"""Auto Configuration Manager ."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import Any, Dict

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
        # Classification
        # Detection
        # Segmentation
        self.task_data_dict = {
            "CLASSIFICATION": ["imagenet"],
            "DETECTION": ["coco", "voc", "yolo"],
            "SEGMENTATION": ["common_semantic_segmentation", "voc", "cityscapes", "ade20k2017", "ade20k2020"],
            #"ACTION_CLASSIFICATION": ["multi-cvat"],
            #"ACTION_DETECTION": ["multi-cvat"],
            #"ANOMALY_CLASSIFICATION": ["mvtec"],
            #"ANOMALY_DETECTION": ["mvtec"],
            #"ANOMALY_SEGMENTATION": ["mvtec"],
            #"INSTANCE_SEGMENTATION": ["coco", "voc"],
        }

    def get_task_type(self, data_format: str) -> str:
        """Detect task type.

        For some datasets (i.e. COCO, VOC, MVTec), can't be fully automated.
        Because those datasets have several format at the same time.
        (i.e. for the COCO case, object detection and instance segmentation annotations coexist)
        In this case, the task_type will be selected to default value.

        For action tasks, currently action_classification is default.

        If Datumaro supports the Kinetics, AVA datasets, MVTec, _is_cvat_format(), _is_mvtec_format()
        functions will be deleted.
        """

        task = ""
        # pick task type
        for task_key, data_value in self.task_data_dict.items():
            if data_format in data_value:
                task = task_key
        return task

    def write_data_with_cfg(self, datasets: Dict[str, IDataset], data_format: str, workspace_dir: str):
        """Save the splitted dataset and data.yaml to the workspace."""
        default_data_folder_name = "dataset"

        data_config = self._create_empty_data_cfg()
        for phase, dataset in datasets.items():
            dst_dir_path = os.path.join(
                workspace_dir,
                default_data_folder_name,
                str(phase),
            )

            # Make data.yaml
            data_config["data"][phase]["data-roots"] = os.path.abspath(dst_dir_path)

            # Convert Datumaro class: DatasetFilter(IDataset) --> Dataset
            datum_dataset = Dataset.from_extractors(dataset)

            # Write the data
            # TODO: consider the way that reduces disk stroage
            # Currently, saving all images to the workspace.
            # It might needs quite large disk storage.
            print(f"[*] Saving {phase} dataset to: {dst_dir_path}")
            DatasetManager.export_dataset(
                dataset=datum_dataset, output_dir=dst_dir_path, data_format=data_format, save_media=True
            )

        self.export_data_cfg(data_config, os.path.join(workspace_dir, "data.yaml"))

    def _create_empty_data_cfg(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Create default dictionary to represent the dataset."""
        # Create empty Data.yaml
        data_subset_format = {"ann-files": None, "data-roots": None}
        data_config = {"data": {subset: data_subset_format.copy() for subset in ("train", "val", "test")}}
        data_config["data"]["unlabeled"] = {"file-list": None, "data-roots": None}
        return data_config

    def export_data_cfg(self, data_cfg: Dict[str, Dict[str, Dict[str, Any]]], output_path: str):
        """Export the data configuration file to output_path."""
        mmcv.dump(data_cfg, output_path)
        print(f"[*] Saving data configuration file to: {output_path}")
